"""Đọc annotation Track 2 (retrieval): ids.csv + queries.csv.

ĐỊNH DẠNG ĐÃ XÁC MINH BẰNG DỮ LIỆU THẬT (không phải suy đoán — test trực tiếp
trên UPAR-Challenge-2027/data/annotations/task2/, 0 lệch trên 20 mẫu ngẫu
nhiên, 30/09/2026):
  - `gt.csv` của task2 GIỐNG HỆT `gt.csv` của task1 (đã diff, byte-identical)
    — dùng thẳng `data.dataset.load_annotations`/`UPARDataset` của Track 1.
  - `queries.csv`: mỗi dòng là MỘT vector 40-bit DUY NHẤT (không trùng lặp
    giữa các dòng theo thiết kế), thứ tự dòng = "semantic id" 0-based.
  - `ids.csv`: mỗi dòng `<image_path>,<semantic_id>` — semantic_id LÀ chỉ số
    dòng 0-based vào `queries.csv`. Đã verify: `queries[ids[image]] ==
    gt[image]` đúng cho mọi ảnh test — tức mỗi ảnh được gán đúng vào nhóm ứng
    với vector nhãn thật của nó.
  - Với một query (một dòng trong queries.csv), tập ảnh "liên quan" (relevant)
    là TẤT CẢ ảnh có cùng semantic_id — tức cùng vector 40-bit tuyệt đối,
    không phải "gần giống". Đây là relevance nhị phân theo exact-match, dùng
    được cho mAP/Rank-1 chuẩn. mADM (metric chính thức của BTC, xem
    code_2/README.md) có thể định nghĩa "liên quan" mềm hơn (theo mức độ
    khớp) — CHƯA có công thức xác nhận, xem README.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


def load_queries(csv_path: str) -> np.ndarray:
    """Trả về (N_query, 40) — mỗi dòng là một vector nhị phân duy nhất."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append([int(float(v)) for v in row])
    return np.array(rows, dtype=np.int64)


def load_ids(csv_path: str) -> dict[str, int]:
    """Trả về dict image_path (tương đối, đúng định dạng trong gt.csv) -> semantic_id."""
    ids = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ids[row[0]] = int(row[1])
    return ids


def build_relevant_groups(ids: dict[str, int]) -> dict[int, list[str]]:
    """Gom ảnh theo semantic_id — trả về dict semantic_id -> list image_path
    có cùng semantic_id đó (tập "liên quan" của query ứng với id này)."""
    groups: dict[int, list[str]] = defaultdict(list)
    for path, sid in ids.items():
        groups[sid].append(path)
    return dict(groups)


def load_retrieval_split(root: str, split_dir: str) -> tuple[np.ndarray, dict[str, int], dict[int, list[str]]]:
    """split_dir: 'annotations/task2/train' hoặc 'annotations/task2/val' (tương
    đối so với `root`). Trả về (queries, ids theo ảnh, nhóm ảnh theo semantic_id).
    """
    queries = load_queries(os.path.join(root, split_dir, "queries.csv"))
    ids = load_ids(os.path.join(root, split_dir, "ids.csv"))
    groups = build_relevant_groups(ids)
    return queries, ids, groups
