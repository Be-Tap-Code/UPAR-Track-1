"""Metric chấm điểm — PHẢI khớp evaluation.md.

    mA       = mean_attr( (TP/(TP+FN) + TN/(TN+FP)) / 2 )
    Label_F1 = mean_attr( F1 của thuộc tính đó )
    Instance_Acc  = mean_image( |P∩G| / |P∪G| )
    Instance_Prec = mean_image( |P∩G| / |P| )   (0 nếu |P|=0, xem ghi chú dưới)
    Instance_Rec  = mean_image( |P∩G| / |G| )   (0 nếu |G|=0)
    Instance_F1   = mean_image( 2·Prec_i·Rec_i / (Prec_i+Rec_i) )   -- QUAN TRỌNG:
        evaluation.md viết Instance_F1 = 2·Prec·Rec/(Prec+Rec) dùng Prec/Rec đã
        trung bình theo ảnh (macro), không phải trung bình các F1 từng ảnh. Ta cung
        cấp cả hai (INSTANCE_F1_MACRO theo công thức macro, và
        instance_f1_mean_per_image nếu cần so sánh) và mặc định dùng công thức
        macro vì đó là công thức ghi trong tài liệu.

    Nếu test gồm nhiều dataset: mỗi dataset chấm riêng trên toàn bộ 40 thuộc tính
    (attribute nào invalid ở dataset đó thì loại khỏi mA/Label_F1 của dataset đó),
    rồi lấy TRUNG BÌNH CỘNG các dataset cho từng metric. Đây là `aggregate_datasets`.

CHƯA XÁC MINH: cách công thức đúng của Instance_Prec/Rec khi |P|=0 hoặc |G|=0
(dấu 0/0). Ta theo quy ước phổ biến trong PAR (Instance_Acc paper gốc UPAR):
trả 1.0 nếu cả P và G đều rỗng (dự đoán đúng "không có gì"), 0.0 nếu chỉ một
bên rỗng. Cần đối chiếu với script gốc của ban tổ chức nếu có thể lấy được.
"""
from __future__ import annotations

import numpy as np


def _binarize(probs: np.ndarray, threshold) -> np.ndarray:
    """threshold: float dùng chung, hoặc array (n_attr,) threshold riêng từng thuộc tính."""
    thr = np.asarray(threshold, dtype=np.float32)
    return (probs > thr).astype(np.int32)


def compute_mA(preds: np.ndarray, targets: np.ndarray, valid_mask: np.ndarray) -> tuple[float, np.ndarray]:
    """preds/targets/valid_mask: (n_samples, n_attr), nhị phân {0,1}.
    valid_mask=0 tại vị trí nhãn ground-truth invalid, bị loại khỏi tính toán.
    Trả về (mA tổng, mA từng thuộc tính) để debug thuộc tính yếu.
    """
    n_attr = preds.shape[1]
    per_attr = np.full(n_attr, np.nan, dtype=np.float64)
    for a in range(n_attr):
        m = valid_mask[:, a].astype(bool)
        if m.sum() == 0:
            continue  # thuộc tính invalid toàn bộ ở dataset này -> loại khỏi mean
        p, t = preds[m, a], targets[m, a]
        pos = t == 1
        neg = t == 0
        tpr = (p[pos] == 1).mean() if pos.sum() > 0 else np.nan
        tnr = (p[neg] == 0).mean() if neg.sum() > 0 else np.nan
        vals = [v for v in (tpr, tnr) if not np.isnan(v)]
        per_attr[a] = float(np.mean(vals)) if vals else np.nan
    return float(np.nanmean(per_attr)), per_attr


def compute_label_f1(preds: np.ndarray, targets: np.ndarray, valid_mask: np.ndarray) -> tuple[float, np.ndarray]:
    n_attr = preds.shape[1]
    per_attr = np.full(n_attr, np.nan, dtype=np.float64)
    for a in range(n_attr):
        m = valid_mask[:, a].astype(bool)
        if m.sum() == 0:
            continue
        p, t = preds[m, a], targets[m, a]
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        denom = 2 * tp + fp + fn
        per_attr[a] = (2 * tp / denom) if denom > 0 else np.nan
    return float(np.nanmean(per_attr)), per_attr


def compute_instance_metrics(preds: np.ndarray, targets: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Trung bình theo ẢNH. valid_mask áp dụng per-image: bỏ cột invalid khỏi P/G
    của ảnh đó trước khi tính giao/hợp — hợp lý vì nhãn invalid nghĩa là "không biết",
    không nên tính là sai cũng không nên tính là đúng.
    """
    n = preds.shape[0]
    acc = np.zeros(n); prec = np.zeros(n); rec = np.zeros(n)
    for i in range(n):
        m = valid_mask[i].astype(bool)
        p = preds[i, m] == 1
        g = targets[i, m] == 1
        inter = int((p & g).sum())
        union = int((p | g).sum())
        p_sum = int(p.sum())
        g_sum = int(g.sum())
        acc[i] = 1.0 if union == 0 else inter / union
        prec[i] = 1.0 if p_sum == 0 and g_sum == 0 else (inter / p_sum if p_sum > 0 else 0.0)
        rec[i] = 1.0 if p_sum == 0 and g_sum == 0 else (inter / g_sum if g_sum > 0 else 0.0)
    inst_acc = float(acc.mean())
    inst_prec = float(prec.mean())
    inst_rec = float(rec.mean())
    denom = inst_prec + inst_rec
    inst_f1 = (2 * inst_prec * inst_rec / denom) if denom > 0 else 0.0
    return {
        "Instance_Acc": inst_acc,
        "Instance_Prec": inst_prec,
        "Instance_Rec": inst_rec,
        "Instance_F1": inst_f1,
    }


def challenge_avg(mA: float, instance_f1: float) -> float:
    denom = mA + instance_f1
    return (2 * mA * instance_f1 / denom) if denom > 0 else 0.0


def evaluate_single_domain(probs: np.ndarray, targets: np.ndarray, valid_mask: np.ndarray,
                            threshold=0.5) -> dict:
    """Chấm một dataset/domain duy nhất. probs: (n, 40) xác suất trong [0,1]."""
    preds = _binarize(probs, threshold)
    mA, mA_per_attr = compute_mA(preds, targets, valid_mask)
    label_f1, label_f1_per_attr = compute_label_f1(preds, targets, valid_mask)
    inst = compute_instance_metrics(preds, targets, valid_mask)
    avg = challenge_avg(mA, inst["Instance_F1"])
    return {
        "Challenge_Avg": avg,
        "mA": mA,
        "Label_F1": label_f1,
        **inst,
        "_mA_per_attr": mA_per_attr,
        "_Label_F1_per_attr": label_f1_per_attr,
    }


def aggregate_datasets(per_dataset_results: list[dict]) -> dict:
    """Trung bình cộng các metric qua nhiều dataset, và std để đo độ ổn định
    (evaluation.md: "The standard deviation over the datasets is reported").
    per_dataset_results: list các dict trả về từ evaluate_single_domain.
    """
    keys = ["Challenge_Avg", "mA", "Label_F1", "Instance_Acc", "Instance_Prec",
            "Instance_Rec", "Instance_F1"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_dataset_results], dtype=np.float64)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out
