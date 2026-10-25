"""Backbone (timm) + pooling + head tuyến tính cho multi-label classification.

Mặc định `vit_large_patch14_clip_224.openai` (CLIP ViT-L/14) — căn cứ: ablation
C²T-Net cho thấy EVA ViT-L đơn lẻ ăn đứt SwinT đơn lẻ trên đúng dòng dữ liệu
UPAR (xem strategy.md mục 3.3 và trao đổi 28/09/2026). Đổi `model.backbone` để
thử ConvNeXt-B (baseline gốc), SigLIP v1, EVA02... rồi so bằng LOO.

Các kỹ thuật generalization bổ sung (xem code/README.md mục "Nghiên cứu bổ
sung" để có bảng đầy đủ + nguồn):
  - `pooling: gem` — GeM pooling thay average pooling (models/pooling.py)
  - `mixstyle.enable: true` — trộn thống kê kênh giữa các ảnh trong batch để
    tổng hợp domain giả lúc train (models/mixstyle.py). CHỈ áp dụng được với
    backbone cho feature map 4D (CNN như ConvNeXt) — với ViT-style token
    sequence, attach_mixstyle_hooks in cảnh báo và bỏ qua thay vì lỗi.
  - `domain_adversarial.enable: true` — Domain-Adversarial Training (DANN,
    models/domain_adversarial.py). Hoạt động trên feature VECTOR đã pool nên
    áp dụng được cho MỌI backbone kể cả ViT/CLIP — đây là kỹ thuật LẤP CHỖ
    TRỐNG mà MixStyle để lại khi dùng backbone không phải CNN.

Tất cả CHƯA được đo trên bài toán này (không có GPU ở máy dev) — chỉ đã xác
minh cú pháp + shape đúng bằng timm thật. BẮT BUỘC so bằng run_loo.py trước
khi giữ lại trong cấu hình cuối.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models.domain_adversarial import DomainClassifier
from models.mixstyle import attach_mixstyle_hooks
from models.pooling import GeM

# Prefix key trong state_dict của domain_classifier — dùng để strip khỏi
# checkpoint lưu ra sau khi train (domain_classifier chỉ cần lúc train, không
# cần lúc eval/export — xem strip_domain_classifier() và train.py).
DOMAIN_CLASSIFIER_PREFIX = "domain_classifier."


# Thứ tự 40 thuộc tính (bản sao của data/attributes.py::ATTRIBUTES; zip nộp KHÔNG có package `data`, nên không import từ đó)
_ATTR_NAMES = ['Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female', 'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald', 'UpperBody-Length-Short', 'UpperBody-Color-Black', 'UpperBody-Color-Blue', 'UpperBody-Color-Brown', 'UpperBody-Color-Green', 'UpperBody-Color-Grey', 'UpperBody-Color-Orange', 'UpperBody-Color-Pink', 'UpperBody-Color-Purple', 'UpperBody-Color-Red', 'UpperBody-Color-White', 'UpperBody-Color-Yellow', 'UpperBody-Color-Other', 'LowerBody-Length-Short', 'LowerBody-Color-Black', 'LowerBody-Color-Blue', 'LowerBody-Color-Brown', 'LowerBody-Color-Green', 'LowerBody-Color-Grey', 'LowerBody-Color-Orange', 'LowerBody-Color-Pink', 'LowerBody-Color-Purple', 'LowerBody-Color-Red', 'LowerBody-Color-White', 'LowerBody-Color-Yellow', 'LowerBody-Color-Other', 'LowerBody-Type-Trousers&Shorts', 'LowerBody-Type-Skirt&Dress', 'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Accessory-Hat']


def strip_domain_classifier(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.startswith(DOMAIN_CLASSIFIER_PREFIX)}


def get_backbone_normalization(backbone_name: str) -> tuple[tuple, tuple]:
    """Mean/std chuẩn hoá ẢNH ĐÚNG theo pretrain gốc của backbone — đọc động từ
    `pretrained_cfg` của timm, KHÔNG dùng chung một bộ ImageNet mean/std cho mọi
    backbone như code từng làm (bug thật, phát hiện 28/09/2026): CLIP ViT-L/14
    pretrain với mean/std KHÁC HẲN ImageNet
    ((0.4815,0.4578,0.4082)/(0.2686,0.2613,0.2758) so với (0.485,0.456,0.406)/
    (0.229,0.224,0.225)) — dùng sai không crash, chỉ âm thầm lệch phân phối
    input so với lúc pretrain, giảm hiệu năng mà khó phát hiện qua test thông
    thường. DINOv3/ConvNeXt tình cờ trùng ImageNet nên không lộ bug trước đó.

    Hàm này chỉ đọc metadata có sẵn trong timm (không tải mạng), nên dùng được
    cả lúc train lẫn lúc inference trong submission.zip (timm đã vendor sẵn).
    Fallback về ImageNet mean/std nếu timm không có pretrained_cfg cho tên này
    (ví dụ backbone tự định nghĩa, không nằm trong registry chuẩn).
    """
    try:
        cfg = timm.get_pretrained_cfg(backbone_name)
        if cfg is not None and cfg.mean is not None and cfg.std is not None:
            return tuple(cfg.mean), tuple(cfg.std)
    except Exception:
        pass
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class PARModel(nn.Module):
    def _load_local_pretrained(self, path: str, img_size) -> None:
        """Nạp trọng số pretrain cục bộ (khoá timm, vd Sapiens từ convert_sapiens.py); nội suy pos_embed sang lưới token mới."""
        from timm.layers import resample_abs_pos_embed
        sd = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        pe = sd.get("pos_embed")
        tgt = self.backbone.pos_embed
        if pe is not None and pe.shape != tgt.shape:
            n_pre = int(getattr(self.backbone, "num_prefix_tokens", 1))
            old = int(round((pe.shape[1] - n_pre) ** 0.5))
            new = tuple(self.backbone.patch_embed.grid_size)
            sd["pos_embed"] = resample_abs_pos_embed(pe, new_size=new, old_size=(old, old), num_prefix_tokens=n_pre)
        res = self.backbone.load_state_dict(sd, strict=False)
        print(f"[pretrained_path] {path}: thiếu={list(res.missing_keys)} thừa={list(res.unexpected_keys)}", flush=True)

    def __init__(self, backbone_name: str, n_attr: int, pretrained: bool = True,
                 drop_rate: float = 0.2, pooling: str = "avg",
                 mixstyle_cfg: dict | None = None, n_domains: int | None = None,
                 grad_checkpointing: bool = False, img_size: tuple | None = None,
                 n_ds_heads: int = 0, drop_path_rate: float = 0.0,
                 text_init: str | None = None, text_scale: float = 50.0,
                 feat_layers: list | None = None, region_head: bool = False,
                 pretrained_path: str | None = None, backbone_kwargs: dict | None = None,
                 region_attn: bool = False):
        super().__init__()
        self.pooling_kind = pooling
        # TUỲ CHỌN (mặc định None = hành vi cũ): kích thước ảnh (H, W) cho ViT — timm tự nội suy position embedding
        # pretrained sang lưới token mới (vd ảnh chữ nhật giữ tỉ lệ người). CNN không dùng tham số này.
        _kw = {"img_size": tuple(img_size)} if img_size else {}
        # TUỲ CHỌN (0 = tắt): stochastic depth của backbone (chuẩn khi fine-tune ViT; trước đây không dùng)
        if drop_path_rate and drop_path_rate > 0:
            _kw["drop_path_rate"] = float(drop_path_rate)
        # TUỲ CHỌN (None = tắt): tham số thêm cho timm.create_model (vd Sapiens: global_pool='avg', fc_norm=False)
        if backbone_kwargs:
            _kw.update(backbone_kwargs)

        self.l2norm = bool(text_init)
        if text_init:
            # [AttriVision-style, 09/10] giữ ĐẦU CHIẾU CLIP (num_classes mặc định = 512) và chuẩn hoá L2; đầu phân loại khởi tạo từ văn bản
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, **_kw)
            self.gem = None
        elif pooling == "gem":
            # global_pool='' -> backbone trả feature CHƯA pool (đã xác minh bằng
            # timm thật: CNN cho (B,C,H,W), ViT cho (B,N,C) token sequence — xem
            # models/pooling.py). Ta tự pool bằng GeM thay vì để timm avg-pool.
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                               num_classes=0, global_pool="", **_kw)
            self.gem = GeM(learn_p=True)
        elif pooling == "avg":
            # num_classes=0 -> timm tự avg-pool (hành vi mặc định theo kiến trúc), trả vector
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, **_kw)
            self.gem = None
        else:
            raise ValueError(f"pooling phải là 'avg' hoặc 'gem', nhận '{pooling}'")

        if pretrained_path:
            self._load_local_pretrained(pretrained_path, img_size)

        feat_dim = self.backbone.num_classes if text_init else self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(feat_dim, n_attr)
        # ĐẶC TRƯNG ĐA TẦNG (tuỳ chọn, feat_layers=[chỉ số khối]): ghép trung bình token của các khối trung gian (LayerNorm riêng) với đặc trưng cuối -> đầu Linear.
        # Giả thuyết: màu/texture (24/40 thuộc tính) nằm ở tầng giữa, bền với dịch miền hơn tầng cuối.
        self.feat_layers = list(feat_layers) if feat_layers else []
        self._taps: dict = {}
        if self.feat_layers:
            self.ml_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in self.feat_layers])
            for _i in self.feat_layers:
                # lưu đầu ra TRÊN CHÍNH KHỐI (không dùng dict của model): bản sao EMA (deepcopy) mới dùng đúng tap của nó
                self.backbone.blocks[_i].register_forward_hook(lambda m, inp, out: setattr(m, '_tap', out))
            self.head = nn.Linear(feat_dim * (1 + len(self.feat_layers)), n_attr)
        # POOLING THEO VÙNG (tuỳ chọn, region_head=True): mỗi thuộc tính dùng đặc trưng pooled trong VÙNG của nó (đầu / thân trên / thân dưới / toàn thân)
        # thay vì trung bình toàn ảnh; mỗi thuộc tính một vector trọng số riêng. Chỉ cho ViT (cần lưới token).
        self.region_mode = bool(region_head)
        if self.region_mode:
            _A = _ATTR_NAMES
            gh, gw = self.backbone.patch_embed.grid_size
            y = (torch.arange(gh).float() + 0.5) / gh
            sg = lambda v: torch.sigmoid(v / 0.04)
            rows = torch.stack([sg(0.28 - y), sg(y - 0.12) * sg(0.62 - y), sg(y - 0.42), torch.ones(gh)])        # (4, gh): đầu, thân trên, thân dưới, toàn thân
            W = rows[:, :, None].expand(4, gh, gw).reshape(4, gh * gw)
            self.register_buffer("region_w", W / W.sum(1, keepdim=True), persistent=False)
            def _g(n):
                if n.startswith(("Hair-", "Accessory-Hat", "Accessory-Glasses")): return 0
                if n.startswith("UpperBody-"): return 1
                if n.startswith("LowerBody-"): return 2
                return 3
            self.register_buffer("region_idx", torch.tensor([_g(n) for n in _A]), persistent=False)
            self.region_weight = nn.Parameter(torch.randn(n_attr, feat_dim) * 0.01)
            self.region_bias = nn.Parameter(torch.zeros(n_attr))
            # ATTENTION THEO THUỘC TÍNH (tuỳ chọn, region_attn=True): trọng số token = softmax(log tiên nghiệm dải + <token, q_a>); q khởi tạo 0 =>
            # đúng bằng pooling theo dải lúc đầu; học dần vị trí theo nội dung (bền hơn khi khung người lệch/rộng hơn lúc train).
            self.attn_q = nn.Parameter(torch.zeros(n_attr, feat_dim)) if region_attn else None
        if text_init:
            import numpy as _np
            from pathlib import Path as _P
            D = _np.load(_P(__file__).resolve().parent.parent / text_init)       # (n_attr, feat_dim), mỗi hàng chuẩn hoá
            assert D.shape == (n_attr, feat_dim), f"text_init {D.shape} != {(n_attr, feat_dim)}"
            with torch.no_grad():
                self.head.weight.copy_(torch.from_numpy(D) * text_scale)
                self.head.bias.zero_()
        # TUỲ CHỌN (n_ds_heads=0 = tắt): head RIÊNG theo dataset dạng phần dư, khởi tạo 0, cộng vào head chung CHỈ lúc train
        # (forward(..., ds_idx=...)); lúc suy luận/validate (ds_idx=None) chỉ dùng head chung nên không đổi gì về đóng gói
        # (bỏ các khoá `ds_heads.` khỏi state_dict khi nộp). Mục đích: để quy ước nhãn/tiên nghiệm riêng của từng dataset bị hấp
        # thụ vào head riêng, còn head chung học phần nhất quán giữa các dataset.
        self.ds_heads = nn.ModuleList()
        for _ in range(n_ds_heads):
            h = nn.Linear(feat_dim, n_attr)
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
            self.ds_heads.append(h)

        if grad_checkpointing:
            # Đánh đổi ~20-30% thời gian/step (recompute activation lúc backward
            # thay vì giữ trong VRAM) để giảm đáng kể bộ nhớ đỉnh — quan trọng với
            # backbone lớn như ViT-L full fine-tune (mọi tham số đều trainable,
            # không freeze). Hầu hết ViT/ConvNeXt trong timm hỗ trợ qua
            # set_grad_checkpointing(); nếu backbone không hỗ trợ, bỏ qua thay vì
            # lỗi cứng (không phải mọi kiến trúc timm đều có API này).
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=True)
            else:
                print(f"[grad_checkpointing] CẢNH BÁO: backbone '{backbone_name}' không hỗ trợ "
                      f"set_grad_checkpointing(), bỏ qua.")

        self.mixstyle_modules = nn.ModuleList()
        if mixstyle_cfg and mixstyle_cfg.get("enable"):
            self.mixstyle_modules = attach_mixstyle_hooks(
                self.backbone,
                layer_names=mixstyle_cfg.get("layers", ["stages.0", "stages.1"]),
                p=mixstyle_cfg.get("p", 0.5),
                alpha=mixstyle_cfg.get("alpha", 0.1),
            )

        # Domain classifier chỉ tồn tại khi n_domains được truyền (train.py tự
        # tính từ số dataset nguồn của fold hiện tại — 2 với fold LOO, 3 với
        # fold "final"). Không tồn tại ở bước inference/nộp bài (n_domains=None
        # mặc định) — submission_template/run.py không cần biết gì về nó.
        self.domain_classifier = None
        if n_domains:
            self.domain_classifier = DomainClassifier(feat_dim, n_domains)

    def _extract_pooled_feat(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.gem is not None:
            feat = self.gem(feat)
        if self.l2norm:
            feat = torch.nn.functional.normalize(feat, dim=-1)
        if self.feat_layers:
            mids = [ln(self.backbone.blocks[i]._tap[:, 1:].mean(1)) for ln, i in zip(self.ml_norms, self.feat_layers)]
            feat = torch.cat([feat] + mids, dim=-1)
        return feat

    def forward(self, x: torch.Tensor, return_features: bool = False, ds_idx: torch.Tensor | None = None):
        """return_features=True: trả thêm feature đã pool (trước dropout/head),
        dùng cho CORAL (models/coral.py) — tránh phải forward backbone 2 lần
        trong cùng 1 bước train. Mặc định False giữ nguyên hành vi cũ (chỉ trả
        logits) để không phá bất kỳ chỗ gọi nào khác (run.py, metrics, EMA...).
        """
        if self.region_mode:
            tok = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:]                # (B, N, C)
            if self.attn_q is not None:
                prior = torch.log(self.region_w[self.region_idx].clamp_min(1e-8))                      # (A, N)
                att = prior[None] + torch.einsum("bnc,ac->ban", tok.float(), self.attn_q.float()) / (tok.shape[-1] ** 0.5)
                reg = torch.einsum("ban,bnc->bac", att.softmax(-1).to(tok.dtype), tok)                  # (B, A, C)
                reg = self.dropout(reg)
            else:
                reg = torch.einsum("rn,bnc->brc", self.region_w.to(tok.dtype), tok)                     # (B, 4, C)
                reg = self.dropout(reg)[:, self.region_idx]                                              # (B, n_attr, C)
            logits = (reg * self.region_weight.to(reg.dtype)).sum(-1) + self.region_bias
            if return_features:
                return logits, reg.mean(1)
            return logits
        feat = self._extract_pooled_feat(x)
        d = self.dropout(feat)
        logits = self.head(d)  # logits, KHÔNG sigmoid ở đây
        if ds_idx is not None and len(self.ds_heads):
            res = torch.stack([h(d) for h in self.ds_heads], dim=1)            # (B, K, n_attr)
            logits = logits + res[torch.arange(res.shape[0], device=res.device), ds_idx]
        if return_features:
            return logits, feat
        return logits

    def forward_with_feat(self, x: torch.Tensor):
        """(logits, feat): feat = đặc trưng (B, D) dùng làm độ tương tự kNN lúc suy luận (không dùng khi train).
        Chế độ vùng: feat = trung bình token (trước pooling theo vùng); chế độ thường: đặc trưng đã pool."""
        if self.region_mode:
            tok = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:]
            reg = torch.einsum("rn,bnc->brc", self.region_w.to(tok.dtype), tok)
            reg = self.dropout(reg)[:, self.region_idx]
            logits = (reg * self.region_weight.to(reg.dtype)).sum(-1) + self.region_bias
            return logits, tok.mean(1)
        feat = self._extract_pooled_feat(x)
        return self.head(self.dropout(feat)), feat

    def forward_with_domain(self, x: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Chỉ dùng lúc train khi bật domain_adversarial. Trả (attr_logits,
        domain_logits). Gradient của domain_logits bị đảo dấu trước khi vào
        backbone (xem models/domain_adversarial.py::gradient_reversal), còn
        gradient của attr_logits truyền bình thường — cả hai cộng vào cùng một
        loss tổng rồi backward MỘT LẦN (không cần 2 bước như SAM).
        """
        if self.domain_classifier is None:
            raise RuntimeError("Model không có domain_classifier — build với n_domains để bật DANN.")
        feat = self._extract_pooled_feat(x)
        attr_logits = self.head(self.dropout(feat))
        domain_logits = self.domain_classifier(feat, lambd)
        return attr_logits, domain_logits


def build_model(cfg: dict, n_attr: int, n_domains: int | None = None) -> PARModel:
    da_cfg = cfg["model"].get("domain_adversarial") or {}
    return PARModel(
        backbone_name=cfg["model"]["backbone"],
        n_attr=n_attr,
        pretrained=cfg["model"].get("pretrained", True),
        drop_rate=cfg["model"].get("drop_rate", 0.2),
        pooling=cfg["model"].get("pooling", "avg"),
        mixstyle_cfg=cfg["model"].get("mixstyle"),
        n_domains=n_domains if da_cfg.get("enable") else None,
        grad_checkpointing=cfg["model"].get("grad_checkpointing", False),
        img_size=cfg["model"].get("img_size"),
        n_ds_heads=int(cfg["model"].get("n_ds_heads", 0)),
        drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.0)),
        text_init=cfg["model"].get("text_init"),
        text_scale=float(cfg["model"].get("text_scale", 50.0)),
        feat_layers=cfg["model"].get("feat_layers"),
        region_head=bool(cfg["model"].get("region_head", False)),
        pretrained_path=cfg["model"].get("pretrained_path"),
        backbone_kwargs=cfg["model"].get("backbone_kwargs"),
        region_attn=bool(cfg["model"].get("region_attn", False)),
    )


def layer_wise_param_groups(model: PARModel, base_lr: float, backbone_lr: float,
                             weight_decay: float = 5e-4, llrd_decay: float | None = None) -> list[dict]:
    """LR nhỏ cho backbone (đã pretrain, không phá feature), LR lớn hơn cho head
    mới + GeM + domain classifier (nếu có). MixStyle không có tham số học được
    nên không xuất hiện ở đây (chỉ ảnh hưởng qua forward hook, không qua optimizer).

    llrd_decay: None hoặc >=1.0 -> hành vi cũ, LR đồng nhất `backbone_lr` cho cả
    backbone (mặc định, tương thích ngược). Nếu trong (0,1): áp Layer-wise LR
    Decay (LLRD) — layer gần input LR nhỏ hơn dần theo cấp số nhân, layer gần
    output LR bằng đúng `backbone_lr`. Căn cứ: literature fine-tune ViT khá
    đồng thuận LLRD thường vượt LR backbone đồng nhất (đề xuất 30/09/2026, xem
    code/README.md). CHỈ áp dụng được khi backbone có `.blocks` (ViT-style
    timm, đã xác minh thật là tên attribute chuẩn — xem models/build.py phía
    trên) — với backbone khác (CNN như ConvNeXt), tự fallback về LR đồng nhất
    + in cảnh báo, không crash.
    """
    head_params = list(model.head.parameters())
    if len(model.ds_heads):
        head_params += list(model.ds_heads.parameters())
    if getattr(model, 'feat_layers', None):
        head_params += list(model.ml_norms.parameters())
    if getattr(model, 'region_mode', False):
        head_params += [model.region_weight, model.region_bias]
        if getattr(model, 'attn_q', None) is not None:
            head_params.append(model.attn_q)
    if model.gem is not None:
        head_params += list(model.gem.parameters())
    if model.domain_classifier is not None:
        head_params += list(model.domain_classifier.parameters())
    head_group = {"params": head_params, "lr": base_lr, "weight_decay": weight_decay}

    use_llrd = llrd_decay is not None and 0 < llrd_decay < 1.0
    if use_llrd and not hasattr(model.backbone, "blocks"):
        print(f"[LLRD] CẢNH BÁO: backbone '{type(model.backbone).__name__}' không có "
              f".blocks (không phải ViT-style timm) — fallback về LR đồng nhất cho backbone.")
        use_llrd = False

    if not use_llrd:
        backbone_params = list(model.backbone.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            head_group,
        ]

    n_blocks = len(model.backbone.blocks)
    block_groups: list[list] = [[] for _ in range(n_blocks)]
    embed_params, final_norm_params = [], []
    for name, p in model.backbone.named_parameters():
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            block_groups[idx].append(p)
        elif name.startswith("norm.") or name.startswith("fc_norm.") or name in ("norm", "fc_norm") or name.startswith("head."):
            final_norm_params.append(p)
        else:  # patch_embed, cls_token, pos_embed, pos_drop, patch_drop, norm_pre...
            embed_params.append(p)

    groups = [{"params": embed_params, "lr": backbone_lr * (llrd_decay ** n_blocks), "weight_decay": weight_decay}]
    for i, params in enumerate(block_groups):
        # block cuối (i = n_blocks-1) -> decay^0 = LR bằng đúng backbone_lr
        lr_i = backbone_lr * (llrd_decay ** (n_blocks - 1 - i))
        groups.append({"params": params, "lr": lr_i, "weight_decay": weight_decay})
    groups.append({"params": final_norm_params, "lr": backbone_lr, "weight_decay": weight_decay})
    groups.append(head_group)
    return groups
