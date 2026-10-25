"""Metric cho Track 2 (attribute-based person retrieval).

CẬP NHẬT 30/09/2026 — công thức mADM CHÍNH XÁC lấy từ trang Codabench Track 2
thật (docs_2/evaluation.md, người dùng dán trực tiếp), không còn là "chưa xác
định" như bản trước. Relevance là EXACT MATCH (all 40 attribute bằng nhau) —
đúng như đã suy ra và verify trên dữ liệu thật trong data/retrieval.py.

Công thức (chép nguyên văn từ evaluation.md, ký hiệu giữ đúng gốc):

Với một query có R "correct match" (ảnh khớp CHÍNH XÁC cả 40 thuộc tính) tại
các rank r_1 < r_2 < ... < r_R trong kết quả xếp hạng của bạn (1-indexed):

    AP  = ( 1/r_1 + 2/r_2 + ... + R/r_R ) / R
    INP = R / r_R

Với dom(g) = (số thuộc tính ảnh g khớp query) / 40, tính cho MỌI ảnh trong
gallery (không chỉ ảnh đã xếp hạng tới), và ndom(g) chuẩn hoá theo mean/max
dom TRÊN TOÀN GALLERY của đúng query đó:

    ndom(g) = max( 0, (dom(g) - mean_dom) / (max_dom - mean_dom) )

    ADM = (1/R) * sum_k [ (ndom(g_1) + ... + ndom(g_{r_k})) / r_k ]

mAP, mADM, mINP = trung bình AP/ADM/INP qua các query. Rank-k = tỷ lệ query
có r_1 <= k (ảnh khớp đầu tiên nằm trong top-k).

Đã verify bằng tay (xem test trong code_2, chạy trực tiếp file này qua
`python -m eval.metrics` để tự kiểm tra lại) trước khi tin dùng cho LOO.
"""
from __future__ import annotations

import numpy as np


def compute_distances(gallery_probs: np.ndarray, query_vecs: np.ndarray,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """L1 distance có TRỌNG SỐ theo thuộc tính giữa xác suất dự đoán (gallery)
    và vector nhị phân (query). weights=None -> đều nhau, khớp NGUYÊN VĂN công
    thức trong baseline mẫu của BTC: sum|q-p| = sum(p) + q.(1-2p).

    Có weights (mục 6.1 docs_2/strategy.md, ý tưởng từ MoSAIC-ReID arXiv
    2512.08697): sum_i w_i*|q_i-p_i| = sum(w*p) + q.(w*(1-2p)) — suy trực tiếp
    từ |q_i-p_i| = p_i + q_i*(1-2*p_i) (q_i nhị phân), nhân thêm w_i rồi cộng
    dồn. Trả về (n_query, n_gallery) — nhỏ hơn = khớp hơn."""
    p = gallery_probs.astype(np.float64)
    q = query_vecs.astype(np.float64)
    w = np.ones(p.shape[1], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    return (w[None, :] * p).sum(axis=1)[None, :] + q @ (w * (1.0 - 2.0 * p)).T


def compute_per_attribute_mA(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """mA từng thuộc tính (balanced accuracy = (TPR+TNR)/2), dùng để ước lượng
    độ tin cậy dự đoán của model cho weights_from_mA bên dưới. KHÔNG dùng
    valid_mask (khác classification_metrics.compute_mA bên Track 1) — nhất
    quán với cách gallery_labels đã được clip 0/1 sẵn ở rank_eval.py/
    run_loo_retrieval.py trước khi truyền vào đây."""
    preds = (probs > threshold).astype(np.int64)
    n_attr = labels.shape[1]
    per_attr = np.full(n_attr, 0.5, dtype=np.float64)
    for a in range(n_attr):
        t, p = labels[:, a], preds[:, a]
        pos, neg = t == 1, t == 0
        tpr = (p[pos] == 1).mean() if pos.sum() > 0 else np.nan
        tnr = (p[neg] == 0).mean() if neg.sum() > 0 else np.nan
        vals = [v for v in (tpr, tnr) if not np.isnan(v)]
        per_attr[a] = float(np.mean(vals)) if vals else 0.5
    return per_attr


def weights_from_mA(mA_per_attr: np.ndarray, min_weight: float = 0.2) -> np.ndarray:
    """Chuyển mA từng thuộc tính (0.5=chance, 1.0=hoàn hảo) thành trọng số cho
    compute_distances: map tuyến tính mA=0.5 -> min_weight, mA=1.0 -> 1.0. Clip
    mA<0.5 về 0.5 trước khi map (mẫu nhỏ có thể cho mA<0.5 do nhiễu, không nên
    suy ra trọng số âm/lớn bất thường). min_weight>0 (mặc định 0.2, không phải
    0) vì MoSAIC-ReID (arXiv 2512.08697) chỉ ra thuộc tính yếu vẫn có tác dụng
    HẠN CHẾ, không phải BẰNG KHÔNG — về 0 tuyệt đối có nguy cơ quá tự tin vào
    ước lượng mA (vốn tính trên val LOO, có nhiễu mẫu)."""
    clipped = np.clip(mA_per_attr, 0.5, 1.0)
    return min_weight + (1.0 - min_weight) * (clipped - 0.5) / 0.5


def _degree_of_match(gallery_labels: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """dom(g) cho MỌI ảnh gallery với MỘT query — (n_gallery,) trong [0,1].
    gallery_labels: (n_gallery, 40) nhãn thật (0/1) của ảnh gallery — LƯU Ý
    đây là NHÃN THẬT, không phải xác suất dự đoán (mADM đo độ khớp thật giữa
    ảnh và query, không phụ thuộc model dự đoán gì — model chỉ quyết định thứ
    tự XẾP HẠNG qua distances, còn dom() luôn tính trên ground truth).
    """
    matches = (gallery_labels == query_vec[None, :]).sum(axis=1)  # số thuộc tính khớp
    return matches / gallery_labels.shape[1]


def evaluate_retrieval(gallery_probs: np.ndarray, gallery_paths: list[str], gallery_labels: np.ndarray,
                        query_vecs: np.ndarray, relevant_groups: dict[int, list[str]],
                        rank_ks: tuple[int, ...] = (1, 5, 10), eps: float = 1e-8,
                        weights: np.ndarray | None = None,
                        distances: np.ndarray | None = None) -> dict:
    """Tính mADM, mAP, Rank-k, mINP trên MỘT gallery (ví dụ 1 dataset LOO).

    gallery_probs: (n_gallery, 40) xác suất dự đoán — quyết định THỨ TỰ xếp hạng.
    gallery_labels: (n_gallery, 40) nhãn THẬT (0/1) — dùng để tính dom()/relevance,
        PHẢI cùng thứ tự với gallery_paths (và gallery_probs).
    relevant_groups: dict semantic_id -> list image_path CÓ semantic_id đó,
        tính trên toàn dataset (data/retrieval.py::build_relevant_groups) — hàm
        này tự lọc lại theo ảnh nào thực sự có mặt trong gallery hiện tại.
    weights: (40,) trọng số theo thuộc tính cho compute_distances (mục 6.1
        docs_2/strategy.md) — None = đều nhau (hành vi baseline cũ, không đổi).
        LƯU Ý: weights PHẢI tính từ dữ liệu KHÁC gallery đang chấm ở đây (ví dụ
        calibrate_attribute_weights.py dùng trung bình 3 fold LOO) — không được
        tính từ chính gallery_labels ở tham số trên, vì gallery ẩn thật lúc nộp
        bài không có nhãn để tự tính trọng số theo kiểu đó.
    """
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    if distances is None:
        distances = compute_distances(gallery_probs, query_vecs, weights=weights)  # (n_query, n_gallery)
    # distances: (n_query, n_gallery), nhỏ hơn = khớp hơn — truyền sẵn để thử cách chấm điểm
    # khác L1 (ví dụ negative log-likelihood) mà không phải sửa hàm này.

    aps, adms, inps, rank_hits = [], [], [], {k: [] for k in rank_ks}
    for qi in range(query_vecs.shape[0]):
        relevant_paths = relevant_groups.get(qi, [])
        relevant_idx = {path_to_idx[p] for p in relevant_paths if p in path_to_idx}
        R = len(relevant_idx)
        if R == 0:
            continue  # không có ảnh khớp trong gallery này -> bỏ qua (không tính vào mẫu số)

        order = np.argsort(distances[qi])  # gần nhất trước, 0-indexed
        rel_mask = np.zeros(distances.shape[1], dtype=bool)
        rel_mask[list(relevant_idx)] = True
        is_relevant = rel_mask[order].astype(np.float64)
        ranks_of_matches = np.where(is_relevant == 1.0)[0] + 1  # 1-indexed, r_1<...<r_R

        # Rank-k
        r1 = ranks_of_matches[0]
        for k in rank_ks:
            rank_hits[k].append(1.0 if r1 <= k else 0.0)

        # AP = (1/R) * sum_k [ k / r_k ]  (k = 1..R, chinh la cum_hits tai r_k)
        ap = float(np.sum((np.arange(1, R + 1)) / ranks_of_matches) / R)
        aps.append(ap)

        # INP = R / r_R
        inp = float(R / ranks_of_matches[-1])
        inps.append(inp)

        # ADM: can dom()/ndom() tren TOAN GALLERY cho query nay
        dom = _degree_of_match(gallery_labels, query_vecs[qi])
        mean_dom, max_dom = dom.mean(), dom.max()
        denom = max_dom - mean_dom
        if denom < eps:
            ndom = np.zeros_like(dom)  # gallery dong deu tuyet doi -> khong phan biet duoc, coi nhu 0
        else:
            ndom = np.clip((dom - mean_dom) / denom, a_min=0.0, a_max=None)
        ndom_ranked = ndom[order]  # sap xep ndom theo dung thu tu xep hang cua model
        cum_ndom = np.cumsum(ndom_ranked)
        adm = float(np.sum(cum_ndom[ranks_of_matches - 1] / ranks_of_matches) / R)
        adms.append(adm)

    n_eval = len(aps)
    out = {
        "mADM": float(np.mean(adms)) if adms else float("nan"),
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "mINP": float(np.mean(inps)) if inps else float("nan"),
        "n_queries_evaluated": n_eval,
        "n_queries_total": int(query_vecs.shape[0]),
    }
    for k in rank_ks:
        out[f"Rank{k}"] = float(np.mean(rank_hits[k])) if rank_hits[k] else float("nan")
    return out
