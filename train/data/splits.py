"""Sinh 3 fold leave-one-dataset-out (LOO) từ 3 dataset nguồn.

UPAR 2027 train/val gồm Market1501, PA100K, PETA. Vì test thật là domain ẩn,
ta mô phỏng domain shift bằng cách: mỗi fold train trên 2 dataset, validate
trên dataset còn lại (giữ nguyên annotation gốc của nó, KHÔNG train trên nó).

Đây là cách chọn model/kỹ thuật đáng tin nhất khi không được đụng vào test thật.
Điểm LOO thấp/cao không phải điểm thi thật, nhưng THỨ TỰ xếp hạng giữa các cấu
hình (config A tốt hơn B hay không) thường đáng tin hơn nhiều so với chỉ nhìn
validation cùng-domain.
"""
from __future__ import annotations

from dataclasses import dataclass

DATASETS = ["Market1501", "PA100k", "PETA"]


@dataclass
class LOOFold:
    name: str
    train_datasets: list[str]
    val_dataset: str


def build_loo_folds(datasets: list[str] = DATASETS) -> list[LOOFold]:
    folds = []
    for held_out in datasets:
        train = [d for d in datasets if d != held_out]
        folds.append(LOOFold(name=f"loo_{held_out}", train_datasets=train, val_dataset=held_out))
    return folds


def build_final_split(datasets: list[str] = DATASETS) -> LOOFold:
    """Cấu hình train bản nộp cuối: train trên TẤT CẢ dataset (train+val gộp,
    nếu luật 2027 cho phép — cần xác nhận, xem strategy.md mục 6). Không có
    val_dataset riêng vì không còn domain nào để giữ lại; theo dõi bằng
    checkpoint tại epoch được chọn từ kết quả LOO thay vì early-stop trên val.
    """
    return LOOFold(name="final", train_datasets=list(datasets), val_dataset=None)
