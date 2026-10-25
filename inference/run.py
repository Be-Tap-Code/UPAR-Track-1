"""[BIẾN THỂ kNN] Entry point nộp lên Codabench. Đúng API theo submission.md:
    load_model() -> None
    predict_batch(samples: list[dict]) -> list[list[float]]   (ưu tiên, nhanh hơn)
    predict_image(sample: dict) -> list[float]                (fallback)

THIẾT KẾ CHỐNG LỖI (sau lần nộp đầu bị "Failed": ingestion không tạo predictions.csv và log
ingestion không xem được): file này KHÔNG ĐƯỢC PHÉP làm sập cả lượt chạy.
  - Ảnh không đọc được / hỏng / bị cắt cụt -> dùng xác suất prior của ban tổ chức cho ảnh đó.
  - Batch lỗi (OOM, lỗi CUDA...) -> thử lại bằng cách chia đôi batch; vẫn lỗi -> prior cho batch đó.
  - load_model() lỗi -> mọi ảnh dùng prior (vẫn ra predictions.csv hợp lệ, điểm thấp thay vì Failed).
  - Mọi lỗi được in ra stderr kèm traceback và đếm lại ở cuối (xem dòng "[run.py] SUMMARY").
Đọc ảnh bằng nhiều luồng (PIL nhả GIL khi giải mã) để chồng lên thời gian tính toán của GPU.

QUAN TRỌNG:
  - Không có mạng lúc chấm. weights.pt, thresholds.json (nếu có), module `models/`, `timm`
    đều nằm sẵn trong thư mục này (export_submission.py lo việc đóng gói).
  - Trả về XÁC SUẤT trong [0,1], đủ 40 giá trị đúng thứ tự attribute_names.
  - threshold 0.5 cố định phía ban tổ chức; nếu có thresholds.json thì dịch bằng bias vào logit.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ảnh JPEG bị cắt cụt vẫn đọc được phần còn lại

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# [09/10] GPU A10 + giới hạn 5 phút: batch lớn hơn và nhiều luồng giải mã hơn (20 nhân CPU); giải mã chunk kế tiếp CHỒNG với tính toán GPU
BATCH_SIZE = int(os.environ.get("UPAR_BATCH", "128"))
NUM_IO_THREADS = min(16, os.cpu_count() or 8)
LOG_EVERY_BATCHES = 20

_state = {
    "model": None, "device": None, "transform": None, "bias": None, "tta_flip": False,
    "amp": False, "prior": {}, "pool": None, "knn": None,
    "n_images": 0, "n_bad_images": 0, "n_fallback_images": 0, "n_batch_errors": 0,
    "n_batches": 0, "t0": None,
}


def _log(msg: str) -> None:
    print(f"[run.py] {msg}", file=sys.stderr, flush=True)


def _bias_from_threshold(thr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Chuyển threshold t thành bias b để cộng vào logit: sigmoid(logit+b) > 0.5
    <=> logit > logit(t) = ln(t/(1-t)). Vậy b = -ln(t/(1-t))."""
    t = np.clip(thr, eps, 1 - eps)
    return -np.log(t / (1 - t))


def _find_asset(name: str) -> str:
    """Tìm file dữ liệu của gói nộp, ưu tiên `assets/` rồi tới gốc archive.

    Submission MẪU của ban tổ chức (examples/task1/sample_code_submission) để dữ
    liệu trong `assets/` — run.py mẫu đọc `assets/attribute_prior.json` và gợi ý
    `assets/model.pth`. Ta đi theo đúng quy ước đó để gói nộp giống mẫu, nhưng vẫn
    chấp nhận file ở gốc để các zip đã đóng gói trước đây không hỏng.
    """
    for p in (os.path.join(_THIS_DIR, "assets", name), os.path.join(_THIS_DIR, name)):
        if os.path.exists(p):
            return p
    return os.path.join(_THIS_DIR, "assets", name)  # để thông báo lỗi chỉ đúng chỗ


def _load_prior() -> dict:
    path = _find_asset("attribute_prior.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return dict(zip(d["attribute_names"], d["positive_rate"]))
    except Exception:
        _log(f"không đọc được {path}, dùng 0.5 làm giá trị dự phòng:\n{traceback.format_exc()}")
        return {}


def _prior_row(names) -> list[float]:
    p = _state["prior"]
    return [float(min(max(p.get(n, 0.5), 0.0), 1.0)) for n in names]


def load_model() -> None:
    _state["prior"] = _load_prior()
    _state["t0"] = time.time()
    _state["pool"] = ThreadPoolExecutor(max_workers=NUM_IO_THREADS)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, _THIS_DIR)
        from model_def import PARModel, get_backbone_normalization

        with open(_find_asset("model_config.json")) as f:
            model_cfg = json.load(f)

        # pooling VÀ img_size PHẢI khớp CHÍNH XÁC lúc train (xem models/build.py):
        # - pooling='gem' thêm submodule có tham số học được.
        # - img_size quyết định shape của pos_embed, mà pos_embed NẰM TRONG
        #   state_dict -> sai img_size là load_state_dict lỗi ngay (hoặc tệ hơn:
        #   load được rồi forward lỗi vì số token không khớp).
        # mixstyle không cần truyền lại: không có tham số học được và tự tắt ở eval().
        model = PARModel(
            backbone_name=model_cfg["backbone"], n_attr=model_cfg["n_attr"],
            pretrained=False, pooling=model_cfg.get("pooling", "avg"),
            img_size=tuple(model_cfg["image_size"]),
            # tham số kiến trúc tuỳ chọn (model_config.json do build_ship_zip.py --model_kw ghi); mặc định không có = hành vi cũ
            **{k: model_cfg[k] for k in ("region_head", "feat_layers", "backbone_kwargs") if k in model_cfg},
        )
        state = torch.load(_find_asset("weights.pt"), map_location="cpu")
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        model.eval().to(device)

        h, w = model_cfg["image_size"]
        mean, std = get_backbone_normalization(model_cfg["backbone"])
        _cn = None
        if model_cfg.get("color_norm"):
            from color_norm import build_color_norm   # color_norm.py nằm ở gốc zip (build_ship_zip.py tự thêm)
            _cn = build_color_norm(model_cfg["color_norm"])
        tf = transforms.Compose(([_cn] if _cn else []) + [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        bias = None
        thr_path = _find_asset("thresholds.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                thr_dict = json.load(f)
            thr = np.array([thr_dict[a] for a in model_cfg["attribute_names"]], dtype=np.float32)
            bias = torch.tensor(_bias_from_threshold(thr), dtype=torch.float32, device=device)

        _state["knn"] = model_cfg.get("knn") or None   # {"k":20,"tau":0.05,"iters":2}: làm mượt xác suất theo láng giềng đặc trưng (tham số cố định, chọn trên LOO)
        _state.update(model=model, device=device, transform=tf, bias=bias,
                      tta_flip=bool(model_cfg.get("tta_flip", False)),
                      amp=bool(model_cfg.get("amp", False)) and device == "cuda")
        _log(f"model ready on {device} (amp={_state['amp']}, tta_flip={_state['tta_flip']}, "
             f"thresholds={'yes' if bias is not None else 'no'}, load={time.time() - _state['t0']:.1f}s)")
    except Exception:
        _state["model"] = None
        _log(f"load_model THẤT BẠI -> dùng prior cho mọi ảnh:\n{traceback.format_exc()}")


def _read_one(sample: dict):
    """Trả về tensor đã chuẩn hoá, hoặc None nếu ảnh không đọc được."""
    path = sample.get("image_path")
    try:
        with Image.open(path) as im:
            return _state["transform"](im.convert("RGB"))
    except Exception as e:
        _state["n_bad_images"] += 1
        _log(f"ảnh không đọc được, dùng prior: {path} ({e!r})")
        return None


@torch.no_grad()
def _forward(x: torch.Tensor):
    """(logits CHƯA cộng bias ngưỡng, feat). Bias ngưỡng áp SAU bước làm mượt (xem predict_batch). Lỗi (vd OOM) thì chia đôi batch."""
    model, device = _state["model"], _state["device"]
    try:
        xb = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=_state["amp"]):
            if hasattr(model, "forward_with_feat") and _state["knn"]:
                lg, ft = model.forward_with_feat(xb)
            else:
                lg, ft = model(xb), None
        return lg.float().cpu(), (ft.float().cpu() if ft is not None else None)
    except Exception:
        if x.shape[0] <= 1:
            raise
        _state["n_batch_errors"] += 1
        _log(f"batch {tuple(x.shape)} lỗi, chia đôi và thử lại:\n{traceback.format_exc()}")
        if device == "cuda":
            torch.cuda.empty_cache()
        h = x.shape[0] // 2
        a, fa = _forward(x[:h]); b_, fb = _forward(x[h:])
        return torch.cat([a, b_], 0), (torch.cat([fa, fb], 0) if fa is not None and fb is not None else None)


def _smooth(probs: np.ndarray, feats: np.ndarray, ok: np.ndarray, k=20, tau=0.05, iters=2, chunk=2048) -> np.ndarray:
    """Làm mượt xác suất từng ảnh bằng k láng giềng gần nhất (cosine) trong các ảnh CÙNG LƯỢT GỌI: p'(g) = sum w p(g'), w = softmax((sim - sim_max)/tau).
    Chỉ dùng NỘI DUNG ảnh (không dùng đường dẫn/thứ tự/kích thước); một bộ tham số cố định cho mọi ảnh."""
    idx = np.where(ok)[0]
    m = len(idx)
    if m < 3:
        return probs
    k = int(min(k, m))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = torch.from_numpy(feats[idx]).to(dev).float()
    F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    Fh = F.half() if dev == "cuda" else F
    I = torch.zeros(m, k, dtype=torch.long, device=dev)
    W = torch.zeros(m, k, device=dev)
    for s0 in range(0, m, chunk):
        S = (Fh[s0:s0 + chunk] @ Fh.T).float()
        sv, si = S.topk(k, dim=1)
        I[s0:s0 + chunk] = si
        W[s0:s0 + chunk] = torch.softmax((sv - sv[:, :1]) / float(tau), dim=1)
    P = torch.from_numpy(probs[idx]).to(dev).float()
    for _ in range(int(iters)):
        P = (W.unsqueeze(-1) * P[I]).sum(1)
    out = probs.copy()
    out[idx] = P.clamp(0, 1).cpu().numpy()
    return out


def predict_batch(samples: list[dict]) -> list[list[float]]:
    if _state["t0"] is None:
        load_model()
    names_per_sample = [s.get("attribute_names") or list(_state["prior"].keys()) for s in samples]
    rows: list[list[float]] = [_prior_row(n) for n in names_per_sample]  # mặc định = prior
    _state["n_images"] += len(samples)

    if _state["model"] is not None:
        # ⚠️ CHIA NHỎ BẮT BUỘC. Ingestion của BTC có thể gọi predict_batch với TOÀN BỘ
        # tập test trong MỘT lần (29,248 ảnh theo log thật "images=29248"). Nếu stack
        # hết một lượt: 29,248 x 3x280x140 x fp32 ~ 13.7 GB + bản sao cho `ok` ~ 27 GB
        # -> trên CPU bị OOM-killer GIẾT (SIGKILL, không phải Exception nên KHÔNG catch
        # được) -> không kịp sinh predictions.csv -> "SCORING FAILED".
        # BATCH_SIZE đã có sẵn trong file nhưng trước đây KHÔNG được dùng ở đây.
        _chunks = [(a, samples[a:a + BATCH_SIZE]) for a in range(0, len(samples), BATCH_SIZE)]
        _prefetch = ThreadPoolExecutor(max_workers=1)   # giải mã chunk kế tiếp trong lúc GPU tính chunk hiện tại

        def _decode(ch):
            return list(_state["pool"].map(_read_one, ch))

        _next = _prefetch.submit(_decode, _chunks[0][1]) if _chunks else None
        n_all = len(samples)
        raw = np.zeros((n_all, len(rows[0])), dtype=np.float32) if n_all else None
        okmask = np.zeros(n_all, dtype=bool)
        feats = None
        for _k, (_s0, _chunk) in enumerate(_chunks):
            _cur = _next
            _next = _prefetch.submit(_decode, _chunks[_k + 1][1]) if _k + 1 < len(_chunks) else None
            try:
                tensors = _cur.result()
                ok = [i for i, t in enumerate(tensors) if t is not None]
                if ok:
                    lg, ft = _forward(torch.stack([tensors[i] for i in ok]))
                    if ft is not None and feats is None:
                        feats = np.zeros((n_all, ft.shape[1]), dtype=np.float16)
                    pr = torch.sigmoid(lg).clamp(0.0, 1.0).numpy()
                    for j, i in enumerate(ok):
                        if pr.shape[1] == raw.shape[1] and np.all(np.isfinite(pr[j])):
                            raw[_s0 + i] = pr[j]; okmask[_s0 + i] = True
                            if ft is not None and np.all(np.isfinite(ft[j].numpy())):
                                feats[_s0 + i] = ft[j].numpy().astype(np.float16)
                        else:
                            _state["n_fallback_images"] += 1
            except Exception:
                _state["n_batch_errors"] += 1
                _state["n_fallback_images"] += len(_chunk)
                _log(f"chunk {_s0} ({len(_chunk)} ảnh) lỗi, dùng prior:\n{traceback.format_exc()}")
            _state["n_batches"] += 1
            if _state["n_batches"] % LOG_EVERY_BATCHES == 0:
                dt = time.time() - _state["t0"]
                _log(f"PROGRESS images={_state['n_images']} elapsed={dt:.0f}s "
                     f"({_state['n_images'] / max(dt, 1e-9):.1f} img/s) "
                     f"bad_images={_state['n_bad_images']} batch_errors={_state['n_batch_errors']} "
                     f"fallback_images={_state['n_fallback_images']}")
        _prefetch.shutdown(wait=False)
        kn = _state.get("knn")
        if kn and feats is not None and okmask.sum() >= 3:
            try:
                t1 = time.time()
                raw = _smooth(raw, feats, okmask, k=kn.get("k", 20), tau=kn.get("tau", 0.05), iters=kn.get("iters", 2))
                _log(f"kNN smoothing (k={kn.get('k', 20)} tau={kn.get('tau', 0.05)} iters={kn.get('iters', 2)}) trên {int(okmask.sum())} ảnh xong trong {time.time() - t1:.1f}s")
            except Exception:
                _log(f"kNN smoothing lỗi -> bỏ qua:\n{traceback.format_exc()}")
        if _state["bias"] is not None and raw is not None:
            b = _state["bias"].cpu().numpy() if hasattr(_state["bias"], "cpu") else np.asarray(_state["bias"])
            if b.shape[0] == raw.shape[1]:
                pc = np.clip(raw, 1e-6, 1 - 1e-6)
                raw = (1.0 / (1.0 + np.exp(-(np.log(pc / (1 - pc)) + b)))).astype(np.float32)
        for i in np.where(okmask)[0]:
            rows[i] = raw[i].tolist()
        _prefetch.shutdown(wait=False)
    else:
        _state["n_fallback_images"] += len(samples)
    return rows


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]


import atexit  # noqa: E402


@atexit.register
def _summary() -> None:
    if _state["t0"] is not None:
        dt = time.time() - _state["t0"]
        _log(f"SUMMARY images={_state['n_images']} elapsed={dt:.0f}s bad_images={_state['n_bad_images']} "
             f"batch_errors={_state['n_batch_errors']} fallback_images={_state['n_fallback_images']}")
