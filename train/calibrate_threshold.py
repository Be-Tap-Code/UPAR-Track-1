"""Tối ưu threshold từng thuộc tính để tối đa hoá ĐÚNG chỉ số xếp hạng (Challenge_Avg).

BỐI CẢNH — vì sao bản cũ (28/09/2026) chưa đủ. Số đo THẬT trên 131,076 ảnh LOO,
chấm đúng cách BTC (mỗi dataset riêng rồi lấy trung bình cộng):

    threshold 0.5 cố định (đúng như BTC chấm)   : Challenge_Avg = 0.8056 +- 0.0217
    ngưỡng tối ưu mA (cách của bản cũ)          : Challenge_Avg = 0.7710 +- 0.0346

Tức là cách cũ KHÔNG chỉ "ít lợi" mà còn TỆ HƠN 3.5 điểm so với không hiệu chỉnh
gì cả. Nguyên nhân: bản cũ grid-search từng thuộc tính độc lập để tối đa hoá **mA**.
mA phân rã được theo thuộc tính nên cách đó tối ưu mA rất tốt (0.7903 -> 0.8358,
+4.6 điểm) — NHƯNG Challenge_Avg là trung bình ĐIỀU HOÀ của mA và Instance_F1, mà
Instance_F1 (tính theo ảnh, trên tập thuộc tính dự đoán P và thật G) KHÔNG phân rã
theo thuộc tính. Đẩy ngưỡng để ăn mA làm Instance_F1 sụp (0.8217 -> 0.7165, -10.5
điểm), và vế sau lấy lại nhiều hơn vế trước đã cho.

Con số 0.7990 -> 0.8080 mà bản cũ in ra là chấm trên tập GỘP 131,076 ảnh — không
phải cách BTC chấm, và bị PA100k (88,987 ảnh, 68% tập gộp) chi phối nên che mất
việc ngưỡng mới tệ đi trên Market1501 và PETA.

BẢN NÀY khác ở ĐÚNG 2 điểm:

  1. Tối ưu trực tiếp Challenge_Avg bằng COORDINATE ASCENT: với mỗi thuộc tính,
     thử từng ngưỡng ứng viên, giữ ngưỡng làm CHỈ SỐ TỔNG lớn nhất, 39 thuộc tính
     khác cố định. Instance_F1 không phân rã được nên không có nghiệm đóng, nhưng
     chỉ có 40 chiều, mỗi chiều là bài toán 1D ⇒ coordinate ascent là cách đúng
     và đủ nhanh. Thử từ CẢ HAI điểm khởi tạo (ngưỡng tối ưu mA và 0.5 đồng loạt)
     rồi giữ kết quả tốt hơn — cần thiết vì leo dốc từ nghiệm tối-ưu-mA phải đi
     qua vùng Challenge_Avg giảm nên rất dễ kẹt ở cực tiểu địa phương.
     (Lưu ý: chỉ DỊCH ngưỡng trong không gian logit mới có nghĩa. Thêm temperature
     scaling KHÔNG giúp gì cho model đơn: sigmoid(a*logit+b) > 0.5 <=> logit > -b/a,
     tức quyết định nhị phân chỉ phụ thuộc tỉ số -b/a nên a là thừa.)

  2. Chấm theo ĐÚNG cách ban tổ chức: mỗi DATASET chấm riêng rồi lấy TRUNG BÌNH
     CỘNG qua dataset (evaluation.md: "Every leaderboard value is the mean over the
     datasets, so a small dataset weighs as much as a large one"). Cần file
     `runs/loo_datasets.npy` do dump_loo_predictions.py sinh ra. Thiếu file đó thì
     script vẫn chạy nhưng cảnh báo rõ là đang chấm trên tập gộp.

Về `--shrink`: ngưỡng tối ưu trên 3 dataset nguồn có thể overfit, mà test là domain
ẩn. Shrink co ngưỡng về 0.5 để bớt overfit. Hệ số shrink KHÔNG chọn bằng cảm tính:
mặc định chọn bằng NESTED leave-one-dataset-out (fit trên 2 dataset, đo trên dataset
thứ 3, xoay vòng) — xem select_shrink_nested. Truyền `--nested_shrink false` để
quay lại hành vi cũ (dùng thẳng --shrink).

TỐC ĐỘ: hàm mục tiêu được gọi ~2,000 lần/vòng, nên KHÔNG được cắt `targets`/`valid`
theo dataset bên trong vòng lặp — bản đầu tiên làm vậy và mỗi lần đánh giá mất 0.57s
(vì `targets[idx]` là int64 (88k,40) = 28MB copy, nhân 3 dataset, nhân 2,000 lần).
build_group_data() tiền xử lý MỘT LẦN: cắt sẵn target/valid và tính sẵn mask bool
+ số đếm. Kết quả: 0.57s -> ~0.03s mỗi lần đánh giá.

Đầu ra `thresholds.json` GIỮ NGUYÊN định dạng cũ ({tên_thuộc_tính: threshold}) để
submission_template/run.py không cần sửa gì.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from data.attributes import ATTRIBUTES, N_ATTR
from eval.metrics import (challenge_avg, compute_instance_metrics, compute_mA,
                          evaluate_single_domain)


# ---------------------------------------------------------------------------
# Tiền xử lý theo dataset
# ---------------------------------------------------------------------------


def build_groups(ds_ids: np.ndarray, n_ds: int) -> list[np.ndarray]:
    """Tách chỉ số dòng theo từng dataset — cơ sở để chấm đúng cách BTC (mỗi
    dataset chấm riêng rồi lấy trung bình cộng, dataset nhỏ ngang dataset lớn)."""
    return [np.flatnonzero(ds_ids == d) for d in range(n_ds)]


def build_group_data(targets: np.ndarray, valid: np.ndarray, groups, names=None) -> list[dict]:
    """Tiền xử lý mỗi dataset MỘT LẦN cho vòng tối ưu:

      idx    : chỉ số dòng của dataset trong mảng gộp (để cắt `preds` mỗi lần chấm)
      targets/valid : đã cắt sẵn (compute_instance_metrics cần nguyên mảng)
      t1, t0 : mask bool (target==1)&valid và (target==0)&valid
      n_pos, n_neg : số mẫu dương/âm hợp lệ của TỪNG thuộc tính (không đổi)

    nhans: chỉ để log cho dễ đọc.
    """
    out = []
    for i, idx in enumerate(groups):
        t = targets[idx]
        v = valid[idx].astype(bool)
        t1 = (t == 1) & v
        t0 = (t == 0) & v
        out.append({
            "name": (names[i] if names else f"ds{i}"),
            "idx": idx,
            "targets": t,
            "valid": v,
            "t1": t1,
            "t0": t0,
            "n_pos": np.count_nonzero(t1, axis=0),
            "n_neg": np.count_nonzero(t0, axis=0),
        })
    return out


def build_candidates(lo: float, hi: float, step: float) -> np.ndarray:
    """Lưới ứng viên ngưỡng, LUÔN chứa đúng 0.5.

    Quan trọng: `arange(0.05, 0.95, 0.02)` KHÔNG chứa 0.5 (nó cho 0.49, 0.51) —
    mà 0.5 chính là nghiệm tốt nhất đo được trên dữ liệu thật (0.8056). Thiếu nó
    thì coordinate ascent bị chặn đúng tại giá trị tốt nhất.
    """
    c = np.round(np.arange(lo, hi + 1e-9, step), 4)
    return np.unique(np.concatenate([c, np.array([0.5], dtype=c.dtype)]))


# ---------------------------------------------------------------------------
# Hàm mục tiêu
# ---------------------------------------------------------------------------


def _group_mA(preds_bool: np.ndarray, gd: dict) -> float:
    """mA của một dataset, vector hoá, TƯƠNG ĐƯƠNG eval/metrics.py::compute_mA.

    Dùng lại mask t1/t0 và số đếm đã tiền xử lý thay vì fancy-index từng thuộc
    tính như compute_mA (compute_mA lặp 40 lần bằng Python — với 2,000 lần đánh
    giá thì riêng phần này đã tốn hàng phút).

    Đẳng thức dùng ở đây: TN = n_neg - FP (đếm bằng số nguyên, không dùng 1 - fp/n
    để tránh lệch 1 ulp so với compute_mA).
    """
    t1, t0 = gd["t1"], gd["t0"]
    tp = np.count_nonzero(preds_bool & t1, axis=0)
    fp = np.count_nonzero(preds_bool & t0, axis=0)
    n_pos, n_neg = gd["n_pos"], gd["n_neg"]
    tn = n_neg - fp
    tpr = np.where(n_pos > 0, tp / np.maximum(n_pos, 1), np.nan)
    tnr = np.where(n_neg > 0, tn / np.maximum(n_neg, 1), np.nan)
    stack = np.stack([tpr, tnr])
    cnt = np.count_nonzero(~np.isnan(stack), axis=0)
    s = np.nansum(stack, axis=0)
    per_attr = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return float(np.nanmean(per_attr))


def score_from_preds(preds_bool: np.ndarray, gds: list[dict]) -> float:
    """Challenge_Avg trung bình cộng qua các dataset — HÀM MỤC TIÊU của tối ưu.
    Nhận mảng nhị phân đã tính sẵn để coordinate ascent cập nhật tăng dần theo
    từng cột (không nhị phân hoá lại toàn bộ (N,40) mỗi lần thử ứng viên).

    Phần Instance dùng ĐÚNG hàm chính thức compute_instance_metrics trên mảng đã
    cắt sẵn — không viết lại, để không có nguy cơ lệch công thức.
    """
    total = 0.0
    for gd in gds:
        pr = preds_bool[gd["idx"]]
        mA = _group_mA(pr, gd)
        inst = compute_instance_metrics(pr, gd["targets"], gd["valid"])
        total += challenge_avg(mA, inst["Instance_F1"])
    return total / max(1, len(gds))


def score_avg(probs, thr, gds: list[dict]) -> float:
    """Bản tiện dụng: nhị phân hoá từ probs rồi chấm (cho các lời gọi lẻ)."""
    return score_from_preds(probs > thr, gds)


def report(probs, targets, valid, thr, groups, names) -> dict:
    """Báo cáo đầy đủ — dùng metric CHÍNH THỨC trong eval/metrics.py (không dùng
    _group_mA) để con số in ra không phụ thuộc bản tối ưu."""
    per_ds = []
    for idx in groups:
        r = evaluate_single_domain(probs[idx], targets[idx], valid[idx], threshold=thr)
        r["_n"] = int(len(idx))
        per_ds.append(r)
    keys = ["Challenge_Avg", "mA", "Label_F1", "Instance_Acc", "Instance_Prec",
            "Instance_Rec", "Instance_F1"]
    agg = {k: float(np.mean([r[k] for r in per_ds])) for k in keys}
    agg["Challenge_Avg_std"] = float(np.std([r["Challenge_Avg"] for r in per_ds]))
    return {"per_dataset": per_ds, "agg": agg, "names": names}


def print_report(tag: str, rep: dict) -> None:
    print(f"--- {tag} ---")
    for name, r in zip(rep["names"], rep["per_dataset"]):
        print(f"    [{name:<12}] n={r['_n']:>6}  Avg={r['Challenge_Avg']:.4f}  "
              f"mA={r['mA']:.4f}  F1={r['Instance_F1']:.4f}  Label_F1={r['Label_F1']:.4f}")
    a = rep["agg"]
    print(f"    {'TRUNG BINH':<14}          Avg={a['Challenge_Avg']:.4f} (+-{a['Challenge_Avg_std']:.4f})  "
          f"mA={a['mA']:.4f}  F1={a['Instance_F1']:.4f}  Label_F1={a['Label_F1']:.4f}  "
          f"Acc={a['Instance_Acc']:.4f}  Prec={a['Instance_Prec']:.4f}  Rec={a['Instance_Rec']:.4f}")


# ---------------------------------------------------------------------------
# Khởi tạo: ngưỡng tối ưu mA (phân rã được theo thuộc tính)
# ---------------------------------------------------------------------------


def _attr_mA_of_group(pred_col: np.ndarray, t: np.ndarray, m: np.ndarray, a: int) -> float:
    """mA của MỘT thuộc tính trên một nhóm."""
    mm = m.astype(bool)
    if mm.sum() == 0:
        return np.nan
    p, tt = pred_col[mm], t[mm]
    pos, neg = tt == 1, tt == 0
    tpr = (p[pos] == 1).mean() if pos.sum() > 0 else np.nan
    tnr = (p[neg] == 0).mean() if neg.sum() > 0 else np.nan
    vals = [v for v in (tpr, tnr) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def search_thresholds_mA(probs, gds: list[dict], candidates) -> np.ndarray:
    """Ngưỡng tối đa hoá mA, tối ưu độc lập từng thuộc tính.

    Hợp lệ vì mA (kể cả mA trung bình qua dataset) PHÂN RÃ theo thuộc tính:
    mean_d mean_a f(d,a) = mean_a mean_d f(d,a).

    CHỈ dùng làm một trong hai ĐIỂM KHỞI TẠO — không dùng làm kết quả cuối. Đo
    trên dữ liệu thật: nghiệm này đạt mA 0.8358 nhưng Instance_F1 chỉ 0.7165,
    Challenge_Avg 0.7710 (tệ hơn cả threshold 0.5). Xem docstring đầu file.
    """
    best_thr = np.full(N_ATTR, 0.5, dtype=np.float32)
    for a in range(N_ATTR):
        col = probs[:, a]
        best_score, best_c = -1.0, 0.5
        for c in candidates:
            pred = (col > c)
            vals = [_attr_mA_of_group(pred[gd["idx"]], gd["targets"][:, a],
                                      gd["valid"][:, a], a) for gd in gds]
            vals = [v for v in vals if not np.isnan(v)]
            score = float(np.mean(vals)) if vals else -1.0
            if score > best_score:
                best_score, best_c = score, float(c)
        best_thr[a] = best_c
    return best_thr


# ---------------------------------------------------------------------------
# Tối ưu trực tiếp Challenge_Avg
# ---------------------------------------------------------------------------


def coordinate_ascent(probs, gds: list[dict], init_thr, candidates,
                      n_sweeps: int = 2, label: str = "", verbose: bool = True) -> tuple[np.ndarray, float]:
    """Leo dốc theo toạ độ trên 40 ngưỡng; hàm mục tiêu = Challenge_Avg trung bình
    qua dataset. Mỗi vòng quét lần lượt 40 thuộc tính, mỗi thuộc tính thử hết
    `candidates`. Dừng sớm nếu một vòng không toạ độ nào cải thiện được.

    Tối ưu tốc độ: giữ mảng nhị phân (N,40) bool và chỉ cập nhật CỘT của thuộc
    tính đang xét, thay vì nhị phân hoá lại toàn bộ mỗi lần thử ứng viên.
    """
    thr = np.array(init_thr, dtype=np.float32).copy()
    preds = probs > thr
    best = score_from_preds(preds, gds)
    tag = f"{label} " if label else ""
    if verbose:
        print(f"  [{tag}init] Challenge_Avg = {best:.4f}", flush=True)
    for sweep in range(n_sweeps):
        n_changed = 0
        for a in range(N_ATTR):
            col = probs[:, a]
            cur = float(thr[a])
            best_c, best_s = cur, best
            for c in candidates:
                c = float(c)
                if c == cur:
                    continue
                preds[:, a] = col > c
                s = score_from_preds(preds, gds)
                if s > best_s + 1e-12:
                    best_s, best_c = s, c
            preds[:, a] = col > best_c
            thr[a] = best_c
            if best_c != cur:
                best = best_s
                n_changed += 1
            if verbose and (a + 1) % 10 == 0:
                print(f"    [sweep {sweep + 1}] {a + 1}/40 thuộc tính, "
                      f"Challenge_Avg hiện tại = {best:.4f}", flush=True)
        if verbose:
            print(f"  [{tag}sweep {sweep + 1}/{n_sweeps}] Challenge_Avg = {best:.4f} "
                  f"({n_changed}/40 thuộc tính đổi ngưỡng)", flush=True)
        if n_changed == 0:
            if verbose:
                print(f"  [{tag}] không còn toạ độ nào cải thiện được, dừng sớm.", flush=True)
            break
    return thr, best


def select_shrink_nested(probs, gds: list[dict], candidates, shrink_grid,
                         nested_sweeps: int = 1, verbose: bool = True) -> tuple[float, dict]:
    """Chọn hệ số shrink bằng nested leave-one-dataset-out.

    Với mỗi dataset bị giữ lại: fit ngưỡng CHỈ trên 2 dataset còn lại, rồi đo với
    từng hệ số shrink trên dataset bị giữ lại (dữ liệu chưa từng thấy khi fit).
    Lấy trung bình qua 3 lần xoay vòng -> chọn shrink tốt nhất.

    Đây là điều kiện để câu "được bao nhiêu điểm" là trung thực: chọn shrink trên
    chính dữ liệu đã dùng để fit ngưỡng thì luôn thiên về shrink nhỏ.
    """
    scores = {float(s): [] for s in shrink_grid}
    for held in range(len(gds)):
        fit_gds = [gds[i] for i in range(len(gds)) if i != held]
        init_mA = search_thresholds_mA(probs, fit_gds, candidates)
        thr_raw, _ = coordinate_ascent(probs, fit_gds, init_mA, candidates,
                                       n_sweeps=nested_sweeps,
                                       label=f"nested/{gds[held]['name']}", verbose=False)
        for s in shrink_grid:
            thr = (1 - float(s)) * thr_raw + float(s) * 0.5
            scores[float(s)].append(score_avg(probs, thr, [gds[held]]))
        if verbose:
            print(f"  [nested] giữ lại {gds[held]['name']:<12} xong", flush=True)

    mean_scores = {s: float(np.mean(v)) for s, v in scores.items()}
    best_s = max(mean_scores, key=lambda k: mean_scores[k])
    if verbose:
        print("  [nested] điểm theo từng shrink (trung bình qua các lần giữ lại):")
        for s in sorted(mean_scores):
            per = ", ".join(f"{x:.4f}" for x in scores[s])
            mark = "  <== CHỌN" if s == best_s else ""
            print(f"      shrink={s:.1f}: mean={mean_scores[s]:.4f}  [{per}]{mark}")
    return best_s, mean_scores


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs_npy", required=True, help="(N,40) probs gộp từ các fold LOO")
    ap.add_argument("--targets_npy", required=True)
    ap.add_argument("--valid_mask_npy", required=True)
    ap.add_argument("--datasets_npy", default=None,
                    help="(N,) chỉ số dataset của từng dòng (dump_loo_predictions.py ghi ra). "
                         "THIẾU file này -> chấm trên tập gộp, KHÔNG phải cách BTC chấm.")
    ap.add_argument("--dataset_names", default=None,
                    help="JSON list tên dataset, chỉ để in báo cáo cho dễ đọc")
    ap.add_argument("--search_range", nargs=2, type=float, default=[0.05, 0.95])
    ap.add_argument("--step", type=float, default=0.02, help="bước ứng viên ngưỡng")
    ap.add_argument("--sweeps", type=int, default=2, help="số vòng coordinate ascent")
    ap.add_argument("--init", choices=["both", "mA", "half"], default="both",
                    help="điểm khởi tạo: both (mặc định) = thử cả hai rồi giữ kết quả tốt hơn")
    ap.add_argument("--shrink", type=float, default=0.3,
                    help="hệ số co ngưỡng về 0.5 (chỉ dùng khi --nested_shrink false)")
    ap.add_argument("--nested_shrink", type=lambda s: str(s).lower() not in ("false", "0", "no"),
                    default=True, help="chọn shrink bằng nested LOO (mặc định: true)")
    ap.add_argument("--nested_sweeps", type=int, default=1,
                    help="số vòng coordinate ascent trong mỗi lần fit của nested")
    ap.add_argument("--eval_subsample", type=int, default=0,
                    help="0 = dùng hết dữ liệu. >0 = chỉ tối ưu trên N dòng (lấy đều mỗi "
                         "dataset, cùng seed) cho nhanh — dùng khi muốn thử nhanh.")
    ap.add_argument("--out", default="thresholds.json")
    args = ap.parse_args()

    t_start = time.time()

    def _elapsed():
        return f"{time.time() - t_start:.0f}s"

    probs = np.load(args.probs_npy)
    targets = np.load(args.targets_npy)
    valid = np.load(args.valid_mask_npy)
    print(f"Đã nạp {probs.shape[0]} ảnh x {probs.shape[1]} thuộc tính", flush=True)

    if args.datasets_npy and os.path.exists(args.datasets_npy):
        ds_ids = np.load(args.datasets_npy)
        n_ds = int(ds_ids.max()) + 1
        groups = build_groups(ds_ids, n_ds)
        if args.dataset_names and os.path.exists(args.dataset_names):
            names = json.load(open(args.dataset_names))
        else:
            names = [f"ds{i}" for i in range(n_ds)]
        print(f"Chấm theo ĐÚNG cách BTC: {n_ds} dataset riêng biệt rồi lấy trung bình "
              f"({', '.join(f'{n}={len(g)}' for n, g in zip(names, groups))})", flush=True)
    else:
        if args.datasets_npy:
            print(f"CẢNH BÁO: không thấy {args.datasets_npy} — quay về chấm trên tập GỘP. "
                  f"Số này KHÔNG so sánh được với điểm leaderboard.", flush=True)
        groups = [np.arange(probs.shape[0])]
        names = ["POOLED"]
        args.nested_shrink = False  # không có dataset nào để giữ lại

    # --- tuỳ chọn lấy mẫu con (chỉ ảnh hưởng TỐI ƯU, báo cáo vẫn chấm toàn bộ) ---
    if args.eval_subsample and args.eval_subsample < len(groups[0]):
        rng = np.random.RandomState(0)
        total_n = sum(len(g) for g in groups)
        fit_groups = []
        for idx in groups:
            take = max(2, int(round(args.eval_subsample * len(idx) / total_n)))
            fit_groups.append(np.sort(rng.choice(idx, size=min(take, len(idx)), replace=False)))
        print(f"eval_subsample={args.eval_subsample} -> tối ưu trên "
              f"{sum(len(g) for g in fit_groups)} dòng; báo cáo vẫn chấm trên TOÀN BỘ "
              f"{total_n} dòng", flush=True)
    else:
        fit_groups = groups

    gds_all = build_group_data(targets, valid, groups, names)
    gds_fit = build_group_data(targets, valid, fit_groups, names)

    candidates = build_candidates(args.search_range[0], args.search_range[1], args.step)
    print(f"Ứng viên ngưỡng: {len(candidates)} giá trị trong "
          f"[{args.search_range[0]}, {args.search_range[1]}] bước {args.step} "
          f"(có chứa 0.5: {0.5 in candidates.tolist()})", flush=True)

    thr_half = np.full(N_ATTR, 0.5, dtype=np.float32)
    print()
    print_report("MỐC SO SÁNH: threshold 0.5 cố định (đúng như BTC chấm)",
                 report(probs, targets, valid, thr_half, groups, names))

    # --- bước 1: điểm khởi tạo ---
    print(f"\n[1/4] Điểm khởi tạo...", flush=True)
    inits: list[tuple[str, np.ndarray, float]] = []
    if args.init in ("both", "mA"):
        print("  tính ngưỡng tối ưu mA (phân rã theo thuộc tính)...", flush=True)
        init_mA = search_thresholds_mA(probs, gds_fit, candidates)
        s_mA = score_avg(probs, init_mA, gds_fit)
        print(f"  -> ngưỡng tối ưu mA: Challenge_Avg = {s_mA:.4f} ({_elapsed()})", flush=True)
        print_report("  (tham khảo) ngưỡng chỉ tối ưu mA",
                     report(probs, targets, valid, init_mA, groups, names))
        inits.append(("mA", init_mA, s_mA))
    if args.init in ("both", "half"):
        s_half = score_avg(probs, thr_half, gds_fit)
        print(f"  -> tất cả 0.5: Challenge_Avg = {s_half:.4f} ({_elapsed()})", flush=True)
        inits.append(("half", thr_half, s_half))
    inits.sort(key=lambda t: -t[2])
    print(f"  => thứ tự khởi tạo (tốt nhất trước): {[n for n, _, _ in inits]}", flush=True)

    # --- bước 2: leo dốc ---
    print(f"\n[2/4] Coordinate ascent ({args.sweeps} vòng, {N_ATTR} thuộc tính "
          f"x {len(candidates)} ứng viên/vòng)...", flush=True)
    best_overall, best_score = None, -1.0
    for name, init, _ in inits:
        thr_raw, sc = coordinate_ascent(probs, gds_fit, init, candidates,
                                        n_sweeps=args.sweeps, label=f"init={name}")
        if sc > best_score:
            best_overall, best_score = thr_raw, sc
    thr_raw = best_overall
    print(f"  => kết quả leo dốc tốt nhất: Challenge_Avg = {best_score:.4f} ({_elapsed()})", flush=True)
    print_report("  ngưỡng thô (chưa shrink)",
                 report(probs, targets, valid, thr_raw, groups, names))

    # --- bước 3: chọn shrink ---
    print(f"\n[3/4] Chọn hệ số shrink... ({_elapsed()})", flush=True)
    if args.nested_shrink and len(groups) > 1:
        print(f"  Chọn bằng nested leave-one-dataset-out ({len(groups)} lần fit)...", flush=True)
        shrink, _ = select_shrink_nested(
            probs, gds_fit, candidates,
            shrink_grid=np.arange(0.0, 0.75, 0.1), nested_sweeps=args.nested_sweeps)
    else:
        shrink = args.shrink
        print(f"  Dùng thẳng --shrink {shrink} (không chọn nested)")
    thr = (1 - shrink) * thr_raw + shrink * 0.5
    print(f"  => shrink = {shrink:.2f} ({_elapsed()})")

    # --- bước 4: kết quả ---
    print()
    print_report("KẾT QUẢ CUỐI: ngưỡng calibrated",
                 report(probs, targets, valid, thr, groups, names))
    print_report("(đối chiếu) ngưỡng thô chưa shrink",
                 report(probs, targets, valid, thr_raw, groups, names))

    with open(args.out, "w") as f:
        json.dump({name: float(t) for name, t in zip(ATTRIBUTES, thr)}, f, indent=2)
    print(f"\nĐã ghi {args.out}  (tổng thời gian {_elapsed()})")
    print("Nhắc: ngưỡng này fit trên 3 domain NGUỒN. Test là domain ẩn — nếu shrink "
          "thắng ở nested LOO thì nên cân nhắc giữ shrink lớn, vì nested LOO vẫn chỉ "
          "đo trên domain nguồn.")


if __name__ == "__main__":
    main()
