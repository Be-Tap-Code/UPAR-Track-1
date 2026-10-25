"""Train một fold (LOO hoặc final). Dùng qua run_loo.py cho việc so sánh cấu hình,
hoặc gọi trực tiếp `python train.py --config ... --fold loo_PETA` / `--fold final`.

CHƯA CHẠY THỬ với dữ liệu thật (máy dev không có GPU/dataset) — trước khi train
dài, chạy nhanh `--epochs 1 --limit_batches 5` để bắt lỗi shape/đường dẫn.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# Fix "RuntimeError: Too many open files" khi num_workers > 0 trên Linux — chiến
# lược mặc định ("file_descriptor") chia sẻ tensor giữa worker process qua file
# descriptor, dễ chạm giới hạn `ulimit -n` của hệ thống khi train nhiều epoch/batch
# (gặp thật trên server: crash giữa lúc validate sau epoch 0, fold lớn nhất —
# 3269 batch train + số batch validate cộng dồn đủ để chạm giới hạn). Đổi sang
# "file_system" (dùng file tạm thay vì file descriptor) — khuyến nghị chính thức
# của PyTorch cho đúng lỗi này, không phụ thuộc `ulimit -n` của server (dễ quên
# set lại mỗi phiên SSH mới). Đặt ở module-level nên chỉ cần import từ đây
# (train.py) là áp dụng luôn cho run_loo.py/dump_loo_predictions.py.
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.dirname(__file__))
from data.attributes import ATTRIBUTES, N_ATTR
from data.dataset import UPARDataset
from data.domain_augment import build_domain_augment_ops
from data.splits import LOOFold, build_final_split, build_loo_folds
from eval.metrics import evaluate_single_domain
from losses.losses import build_loss
from models.build import (build_model, get_backbone_normalization, layer_wise_param_groups,
                           strip_domain_classifier)
from models.coral import multi_domain_coral_loss
from models.domain_adversarial import dann_lambda_schedule
from optim.sam import SAM


def state_dict_to_cpu(state_dict: dict) -> dict:
    """Clone state_dict sang CPU thay vì `copy.deepcopy()` trực tiếp trên GPU
    tensor — bug thật đã gây OOM: deepcopy giữ NGUYÊN state_dict trên GPU, nên
    mỗi lần best_state cập nhật lại có thêm một bản sao đầy đủ tham số model
    NẰM TRÊN VRAM, cộng dồn với model chính + ema_model (nếu bật EMA) + gradient
    + optimizer state của AdamW (2x tham số cho momentum/variance). Với ViT-L
    full fine-tune, riêng phần này đã ngốn thêm hàng GB không cần thiết —
    best_state chỉ cần lưu ra đĩa sau cùng, không cần ở lại GPU.
    """
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def get_amp_context(device: str, enabled: bool):
    """bfloat16 mixed precision — Ampere (RTX 3090 trở lên) hỗ trợ bf16 native qua
    Tensor Core, không cần GradScaler (khác fp16). Giúp giảm VRAM và tăng tốc
    đáng kể cho backbone lớn (ViT-L...). Tự tắt trên CPU (autocast bf16 trên CPU
    chạy được nhưng không có lợi ích tốc độ, giữ enabled=False cho gọn khi debug
    trên máy không GPU).
    """
    if not enabled or device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def build_transforms(cfg):
    h, w = cfg["data"]["image_size"]
    # Mean/std ĐÚNG theo backbone đang dùng, không hardcode ImageNet cho mọi
    # backbone — xem models/build.py::get_backbone_normalization (bug thật:
    # CLIP pretrain với mean/std khác hẳn ImageNet, dùng sai giảm hiệu năng
    # âm thầm, không crash).
    mean, std = get_backbone_normalization(cfg["model"]["backbone"])
    # TUỲ CHỌN: ghi đè mean/std (backbone cục bộ như Sapiens không có pretrained_cfg trong timm)
    if cfg["model"].get("norm_mean") and cfg["model"].get("norm_std"):
        mean, std = tuple(cfg["model"]["norm_mean"]), tuple(cfg["model"]["norm_std"])

    # TUỲ CHỌN data.color_norm ({kind: grayworld, strength: s}): chuẩn hoá màu theo ảnh, áp ở ĐẦU cả train lẫn val (và run.py khi nộp)
    from color_norm import build_color_norm
    _cn = build_color_norm(cfg["data"].get("color_norm"))
    train_ops = ([_cn] if _cn else []) + [transforms.Resize((h, w))]
    if cfg["train"]["augment"].get("random_crop"):
        # rrc_ratio_jitter (mặc định None = hành vi cũ của code/: ratio (0.75..1.33)x(w/h), RỘNG): nếu có (vd 0.15) thì ratio = (w/h)*(1±jitter),
        # đúng công thức của bản nộp 0.7750 (track_1_mq).
        _j = cfg["train"]["augment"].get("rrc_ratio_jitter")
        _ratio = (w / h * (1 - _j), w / h * (1 + _j)) if _j else (0.75 * w / h, (4 / 3) * w / h)
        train_ops.append(transforms.RandomResizedCrop((h, w), scale=(float(cfg["train"]["augment"].get("rrc_scale_min", 0.8)), 1.0), ratio=_ratio))
    if cfg["train"]["augment"].get("horizontal_flip"):
        train_ops.append(transforms.RandomHorizontalFlip())
    # TUỲ CHỌN (15/10): giả lập GÓC NHÌN camera giám sát đặt cao (phối cảnh + co dọc) — chưa từng thử cho Track 1.
    _pv = cfg["train"]["augment"].get("perspective") or {}
    if float(_pv.get("p", 0) or 0) > 0:
        train_ops.append(transforms.RandomPerspective(distortion_scale=float(_pv.get("scale", 0.3)), p=float(_pv["p"])))
    cj = cfg["train"]["augment"].get("color_jitter", 0.0)
    if cj > 0:
        train_ops.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    if cfg["train"]["augment"].get("augmix"):
        train_ops.append(transforms.AugMix())
    # Augmentation giả lập khác biệt camera/domain (blur, nén JPEG, hạ rồi
    # tăng độ phân giải) — xem data/domain_augment.py. Đặt SAU augmix/color
    # nhưng TRƯỚC ToTensor (các op này làm việc trên PIL.Image). Rỗng theo mặc
    # định (train.augment.domain_sim.enable: false) — bật + so LOO trước khi giữ.
    train_ops.extend(build_domain_augment_ops(cfg))
    train_ops.append(transforms.ToTensor())
    train_ops.append(transforms.Normalize(mean=mean, std=std))
    if cfg["train"]["augment"].get("random_erasing", 0) > 0:
        train_ops.append(transforms.RandomErasing(p=cfg["train"]["augment"]["random_erasing"]))
    val_ops = ([_cn] if _cn else []) + [
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def make_dataset(cfg, datasets_filter, transform):
    """Dựng UPARDataset từ các gt.csv cấu hình trong data.gt_csvs (gộp train+val
    của starter kit, xem code/README.md), lọc theo tên dataset nguồn nếu cho.
    """
    root = cfg["data"]["root"]
    csv_paths = [os.path.join(root, p) for p in cfg["data"]["gt_csvs"]]
    return UPARDataset(csv_paths, data_root=root, datasets_filter=datasets_filter, transform=transform)


def make_balanced_sampler(ds: UPARDataset, ds_weights: dict | None = None) -> WeightedRandomSampler:
    """Trọng số nghịch đảo theo số ảnh mỗi dataset, để dataset nhỏ không bị lấn át
    (ví dụ PETA ~15.6k so với PA100k ~89k trong 2 dataset train+val của 1 fold)."""
    counts = {}
    for n in ds.dataset_names:
        counts[n] = counts.get(n, 0) + 1
    # ds_weights (tuỳ chọn): {tên dataset: hệ số}; tỉ lệ kỳ vọng của dataset d trong batch ∝ hệ số (mặc định 1 = cân bằng 50/50)
    ds_weights = ds_weights or {}
    weights = [float(ds_weights.get(n, 1.0)) / counts[n] for n in ds.dataset_names]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_pos_weight(labels: np.ndarray, valid_mask: np.ndarray) -> torch.Tensor:
    pos = (labels * valid_mask).sum(axis=0)
    neg = ((1 - labels) * valid_mask).sum(axis=0)
    pw = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(pw, dtype=torch.float32).clamp(max=50.0)  # clamp để tránh loss nổ với thuộc tính cực hiếm


def build_static_masks(cfg: dict, labels: np.ndarray, dataset_names: list[str]):
    """Mặt nạ TĨNH của loss: (zero_vec theo dataset, hair_idx, static_valid (N,40)). Dùng chung cho train và `--check_masks`.
    CHỐT CHẶN (sau sự cố E7: mask_min_pos=10 che 9 ảnh dương hiếm của PETA -> LowerBody-Orange hết ảnh dương -> mA 0.687 -> 0.475):
    in số nhãn dương hiệu dụng của mọi thuộc tính bị mask đổi; LỖI nếu thuộc tính có nhãn dương gốc mà hiệu dụng = 0."""
    zero_vec = None
    by_ds = cfg["train"].get("mask_attrs_by_dataset") or {}
    if cfg["train"].get("mask_zero_attrs") or by_ds:
        zero_vec = {}
        for name in sorted(set(dataset_names)):
            sel = np.array([n == name for n in dataset_names])
            zero_vec[name] = np.zeros(len(ATTRIBUTES), dtype=bool)
            if cfg["train"].get("mask_zero_attrs"):
                # mask_min_pos (mặc định 1 = chỉ che thuộc tính KHÔNG có nhãn dương nào): che thuộc tính có < mask_min_pos ảnh dương
                # trong dataset đó. CẢNH BÁO: đặt > 1 có thể che nhãn dương hiếm nhưng THẬT (xem docstring).
                zero_vec[name] |= ((labels[sel] == 1).sum(0) < int(cfg["train"].get("mask_min_pos", 1)))
            # mask_attrs_by_dataset: {dataset: [tên thuộc tính]} — che THỦ CÔNG các thuộc tính đã xác minh nhãn của dataset đó xung đột
            for attr in by_ds.get(name, []):
                if attr not in ATTRIBUTES:
                    raise ValueError(f"mask_attrs_by_dataset: không có thuộc tính '{attr}'")
                zero_vec[name][ATTRIBUTES.index(attr)] = True
            print(f"[mask] {name}: che {int(zero_vec[name].sum())} thuộc tính khỏi loss: "
                  f"{[ATTRIBUTES[i] for i in np.where(zero_vec[name])[0]]}", flush=True)

    # TUỲ CHỌN (mặc định tắt): che theo HÀNG nhóm Hair-Length khi cả 3 cột đều = 0 (nhãn thiếu bị lưu thành 0:
    # PA100k 10.7% ảnh, Market1501 3.2%, PETA 0.7%) — không phải "không có tóc".
    hair_idx = None
    if cfg["train"].get("mask_hair_zero_rows"):
        hair_idx = [ATTRIBUTES.index(a) for a in ("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald")]
        print(f"[mask] mask_hair_zero_rows: bật (cột {hair_idx})", flush=True)

    static_valid = (labels != -1).astype(np.float32)
    if zero_vec is not None:
        for name, zv in zero_vec.items():
            sel = np.array([n == name for n in dataset_names])
            static_valid[np.ix_(sel, np.where(zv)[0])] = 0.0
    if hair_idx is not None:
        miss = (labels[:, hair_idx].clip(min=0).sum(1) == 0)
        static_valid[np.ix_(miss, hair_idx)] = 0.0

    # --- chốt chặn: nhãn dương hiệu dụng ---
    orig = (labels == 1).sum(0)
    eff = ((labels == 1) & (static_valid > 0)).sum(0)
    changed = np.where(orig != eff)[0]
    if len(changed):
        print("[mask] nhãn dương hiệu dụng của các thuộc tính bị mask đổi (gốc -> còn lại): "
              + ", ".join(f"{ATTRIBUTES[i]} {int(orig[i])}->{int(eff[i])}" for i in changed), flush=True)
    dead = [ATTRIBUTES[i] for i in range(len(ATTRIBUTES)) if orig[i] > 0 and eff[i] == 0]
    if dead:
        raise ValueError(f"Mask xoá HẾT nhãn dương của {dead} — model không thể học chúng. Sửa mask (vd mask_min_pos) trước khi train.")
    few = [f"{ATTRIBUTES[i]}={int(eff[i])}" for i in changed if eff[i] < 30]
    if few:
        print(f"[mask] CẢNH BÁO: còn rất ít nhãn dương sau mask: {few}", flush=True)
    return zero_vec, hair_idx, static_valid


def train_one_fold(cfg: dict, fold: LOOFold, device: str = "cuda", epochs: int | None = None,
                    limit_batches: int | None = None):
    train_tf, val_tf = build_transforms(cfg)
    train_ds = make_dataset(cfg, fold.train_datasets, transform=train_tf)
    # TUỲ CHỌN (14/10, từ EDA): data.soft_labels = npz {paths, Y (n,40) float} -> nhãn mềm đồng thuận theo ảnh gần trùng (PA100k nhiễu màu);
    # chỉ áp cho tập TRAIN, tập đánh giá giữ nhãn gốc.
    if cfg["data"].get("soft_labels"):
        _z = np.load(cfg["data"]["soft_labels"], allow_pickle=False)
        train_ds.soft_targets = {str(p): y for p, y in zip(_z["paths"], _z["Y"])}
        print(f"[soft_labels] {len(train_ds.soft_targets)} ảnh dùng nhãn mềm ({cfg['data']['soft_labels']})", flush=True)
    # TUỲ CHỌN (14/10): data.mask_attrs = {dataset: [tiền tố thuộc tính]} -> nhãn đó của dataset đó coi là không hợp lệ khi TRAIN
    # (vd bỏ nhãn màu PA100k vì nhiễu: cặp ảnh gần trùng bất đồng màu quần 20%).
    if cfg["data"].get("mask_attrs"):
        train_ds.mask_by_dataset = {dn: np.array([0.0 if a.startswith(tuple(pf)) else 1.0 for a in ATTRIBUTES], np.float32)
                                    for dn, pf in cfg["data"]["mask_attrs"].items()}
        print(f"[mask_attrs] {{{', '.join(f'{k}: {int((v == 0).sum())} thuộc tính' for k, v in train_ds.mask_by_dataset.items())}}}", flush=True)

    sampler = make_balanced_sampler(train_ds, cfg["train"].get("dataset_sampling_weights")) if cfg["train"].get("balanced_dataset_sampling") else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg["data"]["num_workers"], drop_last=True,
    )

    val_loader = None
    if fold.val_dataset is not None:
        val_ds = make_dataset(cfg, [fold.val_dataset], transform=val_tf)
        # TUỲ CHỌN (mặc định tắt): validate trên mẫu CỐ ĐỊNH để thí nghiệm nhanh hơn nhiều
        # (PA100k có 89k ảnh val). Cùng seed 0 -> cùng tập ảnh ở mọi thí nghiệm, so sánh được.
        n_sub = cfg["data"].get("val_subsample")
        if n_sub and n_sub < len(val_ds):
            idx = np.sort(np.random.RandomState(0).choice(len(val_ds), size=int(n_sub), replace=False))
            val_ds = Subset(val_ds, idx)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                 num_workers=cfg["data"]["num_workers"])

    zero_vec, hair_idx, static_valid = build_static_masks(cfg, train_ds.labels, train_ds.dataset_names)
    pos_weight = compute_pos_weight(train_ds.labels.clip(min=0), static_valid)
    # loss.pos_weight_power (mặc định 1.0 = hành vi cũ): pos_weight ** power. <1 làm nhẹ trọng số lớp hiếm (nghiêng về F1/độ chính xác),
    # >1 nặng hơn (nghiêng về mA/recall). Điểm Challenge_Avg là trung bình điều hoà của mA và F1 nên đây là núm chỉnh điểm vận hành.
    pw_pow = float(cfg["loss"].get("pos_weight_power", 1.0))
    if pw_pow != 1.0:
        pos_weight = pos_weight ** pw_pow
    print("[pos_weight] " + ", ".join(f"{a}={float(pos_weight[ATTRIBUTES.index(a)]):.1f}"
                                      for a in ("UpperBody-Color-Pink", "UpperBody-Color-Red", "Accessory-Bag", "Hair-Length-Short")),
          flush=True)

    # DANN và CORAL đều CHỈ LÀ ỨNG VIÊN THỬ NGHIỆM, không phải "cứ bật là lợi":
    # DomainBed (Gulrajani & Lopez-Paz, ICLR 2021) cho thấy ERM (không dùng kỹ
    # thuật DG nào) thường ngang bằng/thắng cả DANN lẫn CORAL khi đánh giá công
    # bằng. Xem models/domain_adversarial.py và models/coral.py cho chi tiết.
    da_cfg = cfg["model"].get("domain_adversarial") or {}
    coral_cfg = cfg["model"].get("coral") or {}
    use_dann = bool(da_cfg.get("enable"))
    use_coral = bool(coral_cfg.get("enable"))
    if use_dann and use_coral:
        raise ValueError("Chưa hỗ trợ bật đồng thời domain_adversarial và coral — chọn một để dễ so sánh LOO.")

    domain_to_idx = {name: i for i, name in enumerate(sorted(fold.train_datasets))}
    n_domains_present = len(domain_to_idx)
    if (use_dann or use_coral) and n_domains_present < 2:
        raise ValueError("DANN/CORAL cần ít nhất 2 dataset nguồn trong fold để có gì đó phân biệt.")
    n_domains = n_domains_present if use_dann else None  # CORAL không cần domain_classifier trong model

    n_ds_heads = int(cfg["model"].get("n_ds_heads", 0))
    if n_ds_heads and (cfg["train"].get("optimizer_kind") == "sam_adamw" or use_dann or use_coral):
        raise ValueError("n_ds_heads mới chỉ hỗ trợ nhánh huấn luyện chuẩn (không SAM/DANN/CORAL).")
    if n_ds_heads:
        from data.splits import DATASETS
        DS_INDEX = {n: i for i, n in enumerate(DATASETS)}
        assert n_ds_heads == len(DATASETS), f"n_ds_heads phải bằng {len(DATASETS)} (số dataset)"
    model = build_model(cfg, N_ATTR, n_domains=n_domains).to(device)
    # loss.use_pos_weight (mặc định true = hành vi cũ): đặt false để dùng BCE thuần (như `bce_ns`), xác suất đã hiệu chuẩn nên
    # ngưỡng 0.5 dùng được; BCE có pos_weight đẩy xác suất lên cao cho thuộc tính hiếm.
    use_pw = cfg["loss"]["type"] == "bce" and cfg["loss"].get("use_pos_weight", True)
    criterion = build_loss(cfg, pos_weight=pos_weight.to(device) if use_pw else None)
    # TUỲ CHỌN (17/10): TRỌNG SỐ LOSS THEO NHÓM THUỘC TÍNH (loss.attr_loss_weight = {tiền tố: hệ số}), vd màu ×2: nhân vào mặt nạ hợp lệ của
    # loss gốc => trung bình có trọng số. Bọc NGAY loss gốc nên các loss phụ (group_ce, inst_loss) vẫn dùng mặt nạ gốc.
    _alw = cfg["loss"].get("attr_loss_weight") or {}
    if _alw:
        _wv = torch.ones(N_ATTR, device=device)
        for _pf, _w in _alw.items():
            for _k, _a in enumerate(ATTRIBUTES):
                if _a.startswith(_pf): _wv[_k] = float(_w)
        _crit0 = criterion
        def criterion(logits, target, valid_mask):
            return _crit0(logits, target, valid_mask * _wv)
        print(f"[attr_loss_weight] {_alw} -> {int((_wv != 1).sum())} thuộc tính", flush=True)
    # TUỲ CHỌN (14/10, từ EDA): CE PHỤ THEO NHÓM LOẠI TRỪ (train.group_ce: {weight, groups}). Nhãn màu áo/quần, tuổi, loại quần là
    # NHÃN ĐƠN (99.8-100% ảnh PA100k/Market đúng 1 lớp) nhưng BCE độc lập không dạy model so sánh các lớp: top-1 màu chỉ đúng 70-75%.
    # Cộng w * CE(softmax trên logit của nhóm, mục tiêu = nhãn nhóm chuẩn hoá; chịu được nhãn mềm của mixup) vào BCE, trên CHÍNH các
    # logit hiện có => suy luận / zip không đổi (vẫn sigmoid + ngưỡng từng thuộc tính).
    _gce = cfg["train"].get("group_ce") or {}
    if float(_gce.get("weight", 0) or 0) > 0:
        _gprefix = _gce.get("groups", ["UpperBody-Color", "LowerBody-Color", "Age-", "LowerBody-Type"])
        _groups = [torch.tensor([k for k, a in enumerate(ATTRIBUTES) if a.startswith(p)], device=device) for p in _gprefix]
        _base_crit, _gw = criterion, float(_gce["weight"])

        def criterion(logits, target, valid_mask):
            loss = _base_crit(logits, target, valid_mask)
            extra, cnt = 0.0, 0
            for g in _groups:
                y, vm = target[:, g].float(), valid_mask[:, g].float()
                s = y.sum(1); ok = (vm.min(1).values > 0) & (s > 0)
                if ok.any():
                    q = y[ok] / s[ok, None]
                    lp = torch.log_softmax(logits[ok][:, g].float(), dim=1)
                    extra = extra + (-(q * lp).sum(1)).mean(); cnt += 1
            return loss + _gw * extra / max(cnt, 1)
        print(f"[group_ce] weight={_gw} groups={_gprefix} sizes={[len(g) for g in _groups]}", flush=True)
    # TUỲ CHỌN (15/10): LOSS PHỤ THEO TỪNG ẢNH nhắm Instance-F1 (nửa còn lại của Challenge_Avg mà BCE theo thuộc tính không tối ưu).
    #   train.inst_loss = {kind: softf1 | zlpr, weight}. softf1: 1 - 2·Σp·y / (Σp + Σy) theo ảnh (p = sigmoid, chịu nhãn mềm).
    #   zlpr (Su 2022): log(1 + Σ_neg e^{s}) + log(1 + Σ_pos e^{-s}) theo ảnh — dương > 0 > âm, ngưỡng tự nhiên ở 0.
    _il = cfg["train"].get("inst_loss") or {}
    if float(_il.get("weight", 0) or 0) > 0:
        _base_crit2, _iw, _ik = criterion, float(_il["weight"]), _il.get("kind", "softf1")

        def criterion(logits, target, valid_mask):
            loss = _base_crit2(logits, target, valid_mask)
            lg, y, vm = logits.float(), target.float(), valid_mask.float()
            if _ik == "softf1":
                p = torch.sigmoid(lg) * vm; yy = y * vm
                extra = (1 - 2 * (p * yy).sum(1) / (p.sum(1) + yy.sum(1)).clamp_min(1e-6)).mean()
            else:
                neg_inf = torch.full_like(lg, -1e4)
                sn = torch.where(vm > 0, lg + torch.log((1 - y).clamp_min(1e-6)), neg_inf)       # trọng số mềm (1-y) cho âm
                sp = torch.where(vm > 0, -lg + torch.log(y.clamp_min(1e-6)), neg_inf)            # trọng số mềm y cho dương
                z = torch.zeros(lg.shape[0], 1, device=lg.device)
                extra = (torch.logsumexp(torch.cat([z, sn], 1), 1) + torch.logsumexp(torch.cat([z, sp], 1), 1)).mean()
            return loss + _iw * extra
        print(f"[inst_loss] kind={_ik} weight={_iw}", flush=True)
    boot_pw =pos_weight.to(device) if use_pw else None   # dùng để quy đổi xác suất về hậu nghiệm thật khi bootstrapping

    param_groups = layer_wise_param_groups(
        model, base_lr=cfg["train"]["lr_head"], backbone_lr=cfg["train"]["lr_backbone"],
        weight_decay=cfg["train"]["weight_decay"], llrd_decay=cfg["train"].get("llrd_decay"),
    )
    optimizer_kind = cfg["train"].get("optimizer_kind", "adamw")
    use_sam = optimizer_kind == "sam_adamw"
    if use_sam and (use_dann or use_coral):
        # Chưa kiểm chứng tương tác giữa 2 bước SAM (nhiễu + update thật) và
        # loss domain phụ — ngữ nghĩa không rõ ràng, cố tình chặn thay vì để
        # chạy ra kết quả không ai hiểu tại sao sai. Chọn một.
        raise ValueError("Chưa hỗ trợ bật đồng thời SAM và domain_adversarial/coral — chọn một.")
    if use_sam:
        # ~2x compute/step (2 forward+backward) — xem cảnh báo trong optim/sam.py
        # về domain generalization trước khi bật mặc định.
        optimizer = SAM(param_groups, torch.optim.AdamW, rho=cfg["train"].get("sam_rho", 0.05))
    else:
        optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)
    amp_ctx = get_amp_context(device, cfg["train"].get("amp", True))
    dann_lambda_max = da_cfg.get("lambda_max", 1.0)
    dann_domain_weight = da_cfg.get("domain_loss_weight", 1.0)
    dann_criterion = nn.CrossEntropyLoss() if use_dann else None
    coral_weight = coral_cfg.get("weight", 1.0)

    # TUỲ CHỌN (mặc định tắt) — CHƯNG CẤT TRI THỨC từ một teacher mạnh hơn (arXiv 2311.01441: student thừa hưởng độ bền OOD
    # của teacher). Loss tổng = loss nhãn cứng + alpha * BCE(student, sigmoid(teacher_logit / T)) * T^2 (che theo valid_mask).
    # Teacher thấy đúng ảnh (đã augment) mà student thấy; teacher_img_size khác student thì nội suy trên GPU.
    kd_cfg = cfg["train"].get("kd") or {}
    kd_on = bool(kd_cfg.get("enable"))
    teacher, kd_alpha, kd_T, kd_size = None, float(kd_cfg.get("alpha", 1.0)), float(kd_cfg.get("T", 1.0)), kd_cfg.get("teacher_img_size")
    if kd_on:
        if use_sam or use_dann or use_coral:
            raise ValueError("Chưng cất (kd) chưa hỗ trợ cùng SAM/DANN/CORAL.")
        tcfg = copy.deepcopy(cfg)
        tcfg["model"].update(backbone=kd_cfg["teacher_backbone"], img_size=kd_size or cfg["model"].get("img_size"),
                             pretrained=False, grad_checkpointing=False, drop_rate=0.0)
        tcfg["model"]["mixstyle"] = {"enable": False}
        tcfg["model"]["domain_adversarial"] = {"enable": False}
        tcfg["model"]["pooling"] = kd_cfg.get("teacher_pooling", "avg")
        teacher = build_model(tcfg, N_ATTR).to(device)
        teacher.load_state_dict(torch.load(kd_cfg["teacher_ckpt"], map_location=device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[kd] teacher {kd_cfg['teacher_backbone']} <- {kd_cfg['teacher_ckpt']} | alpha={kd_alpha} T={kd_T}", flush=True)

    def kd_loss_fn(student_logits, images, valid_mask):
        with torch.no_grad(), amp_ctx:
            x = images if not kd_size else torch.nn.functional.interpolate(
                images, size=tuple(kd_size), mode="bilinear", align_corners=False, antialias=True)
            t_logits = teacher(x).float()
        soft = torch.sigmoid(t_logits / kd_T)
        l = torch.nn.functional.binary_cross_entropy_with_logits(student_logits.float() / kd_T, soft, reduction="none")
        vm = valid_mask.float()
        return (l * vm).sum() / vm.sum().clamp_min(1.0) * (kd_T ** 2)

    # TUỲ CHỌN (mặc định 0 = tắt): LP-FT / warm-start head (ERM++, Kumar et al.) — chỉ huấn luyện head trong lp_steps bước
    # đầu (backbone đóng băng) rồi mới mở backbone, để head ngẫu nhiên không làm méo đặc trưng pretrain.
    lp_steps = int(cfg["train"].get("lp_steps", 0) or 0)
    lp_frozen = False
    if lp_steps > 0:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        lp_frozen = True

    ema_model = None
    if cfg["train"]["ema"]["enable"]:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    def update_ema(decay):
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
            # Buffer (running_mean/var của BatchNorm...) KHÔNG phải tham số: nếu không sao chép thì EMA giữ thống kê BN
            # lúc khởi tạo -> điểm EMA hỏng với model có BN (đã gặp ở MobileCLIP/FastViT). ViT/ConvNeXt (LayerNorm) không bị.
            for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                b_ema.copy_(b)

    n_epochs = epochs or cfg["train"]["epochs"]
    best_avg, best_state = -1.0, None
    history = []
    steps_per_epoch = min(len(train_loader), limit_batches) if limit_batches else len(train_loader)
    total_steps = max(1, n_epochs * steps_per_epoch)
    global_step = 0
    # TUỲ CHỌN train.lr_schedule = "cosine" | "linear" (mặc định None = LR hằng + plateau như cũ): giảm LR theo TỪNG BƯỚC tới 0 trong toàn bộ n_epochs
    # (kèm train.lr_warmup_steps); fine-tune 1 epoch của model L không có giảm LR nên đây là đòn bẩy chưa thử.
    _lrs = cfg["train"].get("lr_schedule")
    lr_sched = None
    if _lrs in ("cosine", "linear"):
        import math as _m
        _wu = int(cfg["train"].get("lr_warmup_steps", 300))
        def _mult(st):
            if st < _wu:
                return (st + 1) / _wu
            pr = min(1.0, (st - _wu) / max(1, total_steps - _wu))
            return 0.5 * (1 + _m.cos(_m.pi * pr)) if _lrs == "cosine" else (1 - pr)
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _mult)
        print(f"[lr_schedule] {_lrs}, warmup {_wu}, total_steps {total_steps}", flush=True)

    # Early stopping: backbone VLM đã pretrain mạnh thường hội tụ sớm khi fine-tune
    # với LR nhỏ (C²T-Net hội tụ ở epoch 4-5/10 với EVA ViT-L — xem strategy.md
    # mục 3.3) — 15 epoch cố định có thể vừa lãng phí GPU-time vừa TỆ HƠN cho đúng
    # mục tiêu domain generalization: train lâu trên 2-3 domain nguồn có nguy cơ
    # overfit vào đặc điểm riêng của các domain đó, phản tác dụng khi test trên
    # domain ẩn. Chỉ áp dụng khi có val_loader (fold LOO) — fold "final" không có
    # domain để validate nên train đủ n_epochs đã chọn từ kết quả LOO.
    early_stop_patience = cfg["train"].get("early_stop_patience", 5)
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for bi, batch in enumerate(tqdm(train_loader, desc=f"[{fold.name}] epoch {epoch}")):
            if limit_batches and bi >= limit_batches:
                break
            if lp_frozen and global_step >= lp_steps:
                for p in model.backbone.parameters():
                    p.requires_grad_(True)
                lp_frozen = False
                print(f"[{fold.name}] hết giai đoạn linear-probe ({lp_steps} bước) -> mở backbone", flush=True)
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            if zero_vec is not None:
                # nhãn "không được gán": thuộc tính không có NHÃN DƯƠNG nào trong dataset đó (vd Market1501 thiếu màu cam/hồng/nâu...)
                # bị che khỏi loss của ảnh thuộc dataset đó, thay vì bị dạy là "âm".
                zm = torch.from_numpy(np.stack([zero_vec[d] for d in batch["dataset"]])).to(device)
                valid_mask = valid_mask * (~zm).to(valid_mask.dtype)
            if hair_idx is not None:
                hair_missing = (target[:, hair_idx].sum(1) == 0).to(valid_mask.dtype)   # (B,) 1 nếu cả 3 cột tóc = 0
                hm = torch.zeros_like(valid_mask)
                hm[:, hair_idx] = 1.0
                valid_mask = valid_mask * (1.0 - hm * hair_missing[:, None])

            if use_sam:
                # Bước 1: gradient tại điểm hiện tại -> nhiễu tham số lên hướng dốc nhất
                with amp_ctx:
                    logits = model(images)
                    loss = criterion(logits, target, valid_mask)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.first_step(zero_grad=True)
                # Bước 2: gradient TẠI điểm đã nhiễu -> update thật, rồi phục hồi tham số gốc
                with amp_ctx:
                    logits2 = model(images)
                    loss2 = criterion(logits2, target, valid_mask)
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.second_step(zero_grad=True)
            elif use_dann:
                # DANN: một forward + một backward duy nhất — gradient reversal
                # (autograd Function, xem models/domain_adversarial.py) đã tự lo
                # việc đảo dấu gradient cho nhánh domain khi backward(), không
                # cần 2 bước thủ công như SAM.
                lambd = dann_lambda_schedule(global_step / total_steps, lambda_max=dann_lambda_max)
                domain_idx = torch.tensor(
                    [domain_to_idx[d] for d in batch["dataset"]], dtype=torch.long, device=device,
                )
                with amp_ctx:
                    logits, domain_logits = model.forward_with_domain(images, lambd)
                    attr_loss = criterion(logits, target, valid_mask)
                    domain_loss = dann_criterion(domain_logits.float(), domain_idx)
                    loss = attr_loss + dann_domain_weight * domain_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            elif use_coral:
                # CORAL: một forward duy nhất, return_features=True để lấy feature
                # đã pool mà không phải forward backbone lần 2 — nhẹ hơn DANN vì
                # không cần domain classifier riêng, chỉ 1 loss phụ cộng thẳng vào.
                domain_idx = torch.tensor(
                    [domain_to_idx[d] for d in batch["dataset"]], dtype=torch.long, device=device,
                )
                with amp_ctx:
                    logits, feat = model(images, return_features=True)
                    attr_loss = criterion(logits, target, valid_mask)
                    c_loss = multi_domain_coral_loss(feat.float(), domain_idx, n_domains_present)
                    loss = attr_loss + coral_weight * c_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            else:
                ds_idx = None
                if n_ds_heads:
                    ds_idx = torch.tensor([DS_INDEX[d] for d in batch["dataset"]], dtype=torch.long, device=device)
                # MIXUP (tuỳ chọn, train.mixup.alpha>0): trộn ảnh và nhãn mềm với một ảnh khác trong batch; vì batch lấy cân bằng giữa các dataset nên
                # nhiều cặp là LIÊN MIỀN (kỹ thuật tổng quát hoá miền kinh điển). Mặt nạ hợp lệ = giao của hai ảnh.
                # REGION-SWAP (tuỳ chọn, train.regionmix.prob>0): ghép NỬA TRÊN ảnh i với NỬA DƯỚI ảnh j (đường cắt ~hông, 45–55% chiều cao) => tổ hợp trang phục mới;
                # nhãn nửa trên (tuổi/giới/tóc/mũ/kính/áo) lấy từ i, LowerBody-* từ j, túi/balo (toàn thân, mơ hồ) bị che. Bước này không dùng mixup.
                _rm = cfg["train"].get("regionmix") or {}
                _swapped = False
                if float(_rm.get("prob", 0)) > 0 and np.random.rand() < float(_rm["prob"]):
                    _H = images.shape[2]
                    _r = int(round(_H * float(np.random.uniform(0.45, 0.55))))
                    _perm = torch.randperm(images.shape[0], device=images.device)
                    images = torch.cat([images[:, :, :_r], images[_perm][:, :, _r:]], dim=2)
                    _lo = torch.tensor([n.startswith("LowerBody-") for n in ATTRIBUTES], device=images.device)
                    _gl = torch.tensor([n.startswith(("Accessory-Backpack", "Accessory-Bag")) for n in ATTRIBUTES], device=images.device)
                    target = torch.where(_lo, target[_perm], target)
                    valid_mask = torch.where(_lo, valid_mask[_perm], valid_mask)
                    valid_mask = valid_mask * (~_gl).to(valid_mask.dtype)
                    _swapped = True
                _mx = cfg["train"].get("mixup") or {}
                if (not _swapped) and float(_mx.get("alpha", 0)) > 0 and np.random.rand() < float(_mx.get("prob", 1.0)):
                    _lam = float(np.random.beta(_mx["alpha"], _mx["alpha"]))
                    _perm = torch.randperm(images.shape[0], device=images.device)
                    if _mx.get("cross_domain"):        # 15/10: ép cặp LIÊN MIỀN (nội suy giữa các dataset nguồn) khi batch có >1 dataset
                        _dsn = list(batch["dataset"]); _B = len(_dsn); _pp = []
                        for _i in range(_B):
                            _c = [j for j in range(_B) if _dsn[j] != _dsn[_i]]
                            _pp.append(int(np.random.choice(_c)) if _c else int(_perm[_i]))
                        _perm = torch.tensor(_pp, device=images.device)
                    _mlayers = _mx.get("manifold_layers")
                    if _mlayers:                         # 15/10: MANIFOLD MIXUP — trộn token ở một block ngẫu nhiên thay vì trộn pixel
                        _blk = model.backbone.blocks[int(np.random.choice(_mlayers))]
                        _hk = _blk.register_forward_hook(lambda m, i, o, _l=_lam, _p=_perm: _l * o + (1 - _l) * o[_p])
                        _manifold_hook = _hk
                    else:
                        images = _lam * images + (1 - _lam) * images[_perm]
                    target = _lam * target + (1 - _lam) * target[_perm]
                    valid_mask = valid_mask * valid_mask[_perm]
                with amp_ctx:
                    logits = model(images, ds_idx=ds_idx)
                    if "_manifold_hook" in locals() and _manifold_hook is not None:
                        _manifold_hook.remove(); _manifold_hook = None
                    # SOFT BOOTSTRAPPING (tuỳ chọn, loss.bootstrap_beta<1): nhãn nhiễu/xung đột giữa dataset -> đích = β·nhãn + (1-β)·dự đoán (không gradient);
                    # bắt đầu sau loss.bootstrap_start_epoch để dự đoán đủ tốt.
                    _bb = float(cfg["loss"].get("bootstrap_beta", 1.0))
                    if _bb < 1.0 and epoch >= int(cfg["loss"].get("bootstrap_start_epoch", 1)):
                        _pp = torch.sigmoid(logits.detach().float())
                        if boot_pw is not None:
                            # BCE có pos_weight làm xác suất đầu ra bị đẩy lên (p* = pw·q/(pw·q+1-q)); dùng thẳng p làm nhãn mềm tạo vòng phản hồi dương
                            # (đã gặp: Avg sụp 0.50). Quy đổi ngược về hậu nghiệm thật q = p/(p+pw(1-p)) trước khi trộn.
                            _pp = _pp / (_pp + boot_pw * (1 - _pp))
                        target = _bb * target + (1 - _bb) * _pp
                    loss = criterion(logits, target, valid_mask)
                    if teacher is not None:
                        loss = loss + kd_alpha * kd_loss_fn(logits, images, valid_mask)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            if ema_model is not None:
                ema_decay = cfg["train"]["ema"]["decay"]
                if cfg["train"]["ema"].get("warmup"):
                    # TUỲ CHỌN (mặc định tắt): với decay 0.9998, EMA còn ~77% trọng số khởi tạo
                    # (cả head ngẫu nhiên) sau 1 epoch ngắn -> điểm eval lệch mạnh ở run ngắn.
                    ema_decay = min(ema_decay, (1 + global_step) / (10 + global_step))
                update_ema(ema_decay)
            if lr_sched is not None:
                lr_sched.step()
            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(1, min(len(train_loader), limit_batches or len(train_loader)))
        eval_model = ema_model if ema_model is not None else model

        tta_flip = cfg["train"].get("tta_flip_eval", False)
        if val_loader is not None:
            metrics = run_validation(eval_model, val_loader, device, limit_batches=limit_batches,
                                      tta_flip=tta_flip, amp=cfg["train"].get("amp", True))
            scheduler.step(1 - metrics["Challenge_Avg"])  # plateau trên -Challenge_Avg thay vì val loss
            print(f"[{fold.name}] epoch {epoch} train_loss={avg_loss:.4f} "
                  f"Avg={metrics['Challenge_Avg']:.4f} mA={metrics['mA']:.4f} F1={metrics['Instance_F1']:.4f}")
            hist = {"epoch": epoch, "train_loss": avg_loss, "Avg": metrics["Challenge_Avg"],
                    "mA": metrics["mA"], "F1": metrics["Instance_F1"]}
            if cfg["train"].get("eval_raw") and ema_model is not None:
                # TUỲ CHỌN (mặc định tắt): chấm thêm model KHÔNG-EMA để tách "model học kém"
                # khỏi "EMA chưa bắt kịp".
                raw = run_validation(model, val_loader, device, limit_batches=limit_batches,
                                     tta_flip=tta_flip, amp=cfg["train"].get("amp", True))
                print(f"[{fold.name}] epoch {epoch} (no-EMA) Avg={raw['Challenge_Avg']:.4f} "
                      f"mA={raw['mA']:.4f} F1={raw['Instance_F1']:.4f}")
                hist.update({"raw_Avg": raw["Challenge_Avg"], "raw_mA": raw["mA"], "raw_F1": raw["Instance_F1"]})
            history.append(hist)
            if metrics["Challenge_Avg"] > best_avg:
                best_avg = metrics["Challenge_Avg"]
                # strip domain_classifier: chỉ cần lúc train (DANN), không cần lúc
                # eval/export — nếu để lẫn, dump_loo_predictions.py/export_submission.py
                # sẽ load_state_dict() lỗi vì model dựng lại ở đó không có domain_classifier.
                best_state = strip_domain_classifier(state_dict_to_cpu(eval_model.state_dict()))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    print(f"[{fold.name}] dừng sớm ở epoch {epoch} — "
                          f"{early_stop_patience} epoch liên tiếp không tăng Challenge_Avg "
                          f"(best={best_avg:.4f})")
                    break
        else:
            print(f"[{fold.name}] epoch {epoch} train_loss={avg_loss:.4f} (no val_dataset, final fold)")
            best_state = strip_domain_classifier(state_dict_to_cpu(eval_model.state_dict()))

        if cfg["train"].get("save_all_epochs"):
            # Để dùng với swa_average.py sau này — trung bình vài epoch cuối
            # (xem swa_average.py, kỹ thuật khác EMA, có cơ sở riêng cho domain
            # generalization). Tắt mặc định vì tốn đĩa nếu train nhiều epoch.
            ep_path = os.path.join(cfg["output_dir"], f"{fold.name}_epoch{epoch}.pt")
            torch.save(strip_domain_classifier(eval_model.state_dict()), ep_path)

    return {"best_avg": best_avg, "state_dict": best_state, "model": model, "history": history}


@torch.no_grad()
def run_validation(model, val_loader, device, limit_batches=None, return_raw=False,
                    tta_flip: bool = False, amp: bool = True):
    """tta_flip: nếu True, trung bình xác suất giữa ảnh gốc và ảnh lật ngang
    (test-time augmentation rẻ nhất, gấp đôi thời gian inference). Dùng để SO
    SÁNH trong LOO trước — bật ở đây không tự động bật khi nộp bài, xem
    submission_template/run.py và export_submission.py cho phần inference thật.
    """
    model.eval()
    amp_ctx = get_amp_context(device, amp)
    all_probs, all_targets, all_valid = [], [], []
    for bi, batch in enumerate(val_loader):
        if limit_batches and bi >= limit_batches:
            break
        images = batch["image"].to(device)
        with amp_ctx:
            logits = model(images)
        probs = torch.sigmoid(logits.float())
        if tta_flip:
            with amp_ctx:
                logits_flip = model(torch.flip(images, dims=[3]))
            probs = (probs + torch.sigmoid(logits_flip.float())) / 2
        probs = probs.cpu().numpy()
        all_probs.append(probs)
        all_targets.append(batch["target"].numpy())
        all_valid.append(batch["valid_mask"].numpy())
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    valid = np.concatenate(all_valid, axis=0)
    metrics = evaluate_single_domain(probs, targets, valid, threshold=0.5)
    if return_raw:
        return metrics, probs, targets, valid
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="all", help="loo_<Dataset> | final | all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_batches", type=int, default=None, help="debug: giới hạn số batch/epoch")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="ghi đè config theo đường dẫn chấm, vd --set loss.type=bce --set train.llrd_decay=null "
                         "(giá trị parse bằng YAML). Lặp lại được.")
    ap.add_argument("--tag", default=None,
                    help="tên thí nghiệm: checkpoint lưu thành <tag>_<fold>.pt thay vì <fold>.pt, "
                         "và kết quả ghi thêm vào <output_dir>/exp_results.jsonl")
    ap.add_argument("--seed", type=int, default=None, help="ghi đè cfg['seed']")
    ap.add_argument("--check_masks", action="store_true", help="chỉ dựng mặt nạ cho các fold rồi thoát (kiểm tra, không train)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for ov in args.overrides:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    if args.fold == "final":
        folds = [build_final_split()]
    elif args.fold == "all":
        folds = build_loo_folds()
    elif args.fold.startswith("src_"):
        # Fold tuỳ ý: src_<A>[+<B>]_to_<C> = train trên A (và B), chấm trên C (vd src_PETA_to_PA100k). Dùng để đo đóng góp
        # biên của một dataset (vd Market1501 có giúp PA100k/PETA không) ngay trong LOO, không cần lượt nộp.
        import re
        m = re.fullmatch(r"src_([A-Za-z0-9+]+)_to_([A-Za-z0-9]+)", args.fold)
        if not m:
            raise ValueError(f"Tên fold không đúng dạng src_A[+B]_to_C: {args.fold}")
        tr, va = m.group(1).split("+"), m.group(2)
        from data.splits import DATASETS
        bad = [d for d in tr + [va] if d not in DATASETS]
        if bad or va in tr:
            raise ValueError(f"Fold {args.fold}: dataset không hợp lệ {bad} hoặc trùng giữa train và val")
        folds = [LOOFold(name=args.fold, train_datasets=tr, val_dataset=va)]
    else:
        folds = [f for f in build_loo_folds() if f.name == args.fold]
        if not folds:
            raise ValueError(f"Unknown fold {args.fold}")

    if args.check_masks:
        for fold in folds:
            print(f"=== {fold.name}: train {fold.train_datasets}", flush=True)
            ds_ = make_dataset(cfg, fold.train_datasets, transform=None)
            build_static_masks(cfg, ds_.labels, ds_.dataset_names)
        print("check_masks: OK")
        return

    for fold in folds:
        result = train_one_fold(cfg, fold, device=device, epochs=args.epochs, limit_batches=args.limit_batches)
        stem = f"{args.tag}_{fold.name}" if args.tag else fold.name
        ckpt_path = os.path.join(cfg["output_dir"], f"{stem}.pt")
        torch.save(result["state_dict"], ckpt_path)
        print(f"Saved {ckpt_path} (best_avg={result['best_avg']:.4f})")
        if args.tag:
            with open(os.path.join(cfg["output_dir"], "exp_results.jsonl"), "a") as f:
                f.write(json.dumps({"tag": args.tag, "fold": fold.name, "seed": seed,
                                    "overrides": args.overrides, "best_avg": result["best_avg"],
                                    "history": result["history"]}) + "\n")


if __name__ == "__main__":
    main()
