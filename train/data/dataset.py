"""UPARDataset: đọc annotation CSV thật của starter kit 2027
(speckean/UPAR-Challenge-2027, đã clone vào ../UPAR-Challenge-2027).

ĐỊNH DẠNG THẬT (đã xác minh bằng cách đọc file thật, không còn là giả định):
  - data/annotations/task1/train/gt.csv và .../val/gt.csv
  - Header dòng đầu bắt đầu bằng "#": "# image,Age-Young,...,Accessory-Hat"
  - Mỗi dòng: <path>,<40 giá trị 0/1>
  - path đã có PREFIX TÊN DATASET: "Market1501/bounding_box_train/xxx.jpg",
    "PA100k/release_data/release_data/xxx.jpg", "PETA/images/xxx.png"
    -> CHỈ CẦN MỘT data_root (thư mục "data/" sau khi chạy download_datasets.py),
       KHÔNG cần images_root riêng từng dataset.
  - Nhãn trong file train/val hiện tại CHỈ có giá trị 0/1, không thấy giá trị
    invalid. evaluation.md nói nhãn invalid có thể xuất hiện ở TEST ẨN theo
    từng dataset, nên ta vẫn giữ cơ chế mask (INVALID_VALUE) để an toàn, nhưng
    trên train/val hiện tại, valid_mask sẽ luôn là toàn 1.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.attributes import ATTRIBUTES, N_ATTR

INVALID_VALUE = -1  # phòng trường hợp test ẩn dùng mã này cho nhãn invalid


def dataset_of(image_path: str) -> str:
    """Tên dataset nguồn suy từ segment đầu tiên của đường dẫn, ví dụ
    "Market1501/bounding_box_train/x.jpg" -> "Market1501"."""
    return image_path.split("/", 1)[0]


def _parse_row(row: list[str]) -> tuple[str, np.ndarray]:
    img_path = row[0]
    vals = np.array([int(float(v)) for v in row[1:1 + N_ATTR]], dtype=np.int64)
    if len(vals) != N_ATTR:
        raise ValueError(f"Expected {N_ATTR} attribute values, got {len(vals)} for row {row[0]!r}")
    return img_path, vals


def load_annotations(csv_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Trả về (list đường dẫn ảnh tương đối, mảng nhãn (N, 40))."""
    image_paths, all_vals = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            img, vals = _parse_row(row)
            image_paths.append(img)
            all_vals.append(vals)
    return image_paths, np.stack(all_vals, axis=0)


class UPARDataset(Dataset):
    """Gộp một hoặc nhiều file gt.csv (ví dụ train + val cùng lúc để có nhiều dữ
    liệu hơn cho LOO), rồi LỌC theo dataset nguồn nếu `datasets_filter` được cho.

    csv_paths: list đường dẫn tới gt.csv (tuyệt đối hoặc tương đối so với cwd)
    data_root: thư mục "data/" chứa Market1501/, PA100k/, PETA/ — ảnh = data_root/path
    datasets_filter: None = giữ tất cả; hoặc list tên dataset cần giữ, ví dụ
        ["Market1501", "PA100k"] khi PETA là dataset bị giữ lại (held-out) của fold LOO
    """

    def __init__(self, csv_paths: list[str | Path], data_root: str | Path,
                 datasets_filter: list[str] | None = None, transform=None):
        self.transform = transform
        self.data_root = str(data_root)
        rel_paths: list[str] = []
        labels_chunks = []
        for csv_path in csv_paths:
            paths, labels = load_annotations(csv_path)
            rel_paths.extend(paths)
            labels_chunks.append(labels)
        all_labels = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.zeros((0, N_ATTR), dtype=np.int64)

        if datasets_filter is not None:
            keep = [i for i, p in enumerate(rel_paths) if dataset_of(p) in datasets_filter]
        else:
            keep = list(range(len(rel_paths)))

        self.rel_paths = [rel_paths[i] for i in keep]
        self.dataset_names = [dataset_of(p) for p in self.rel_paths]
        self.labels = all_labels[keep] if len(keep) else all_labels
        assert len(self.rel_paths) == len(self.labels), "Số ảnh và số dòng nhãn lệch nhau"

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_root, self.rel_paths[idx])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        raw = self.labels[idx]
        valid_mask = (raw != INVALID_VALUE).astype(np.float32)
        target = np.where(raw == INVALID_VALUE, 0, raw).astype(np.float32)
        soft = getattr(self, "soft_targets", None)          # TUỲ CHỌN (chỉ gắn cho tập TRAIN): nhãn mềm đồng thuận thay nhãn gốc
        if soft is not None and self.rel_paths[idx] in soft:
            target = soft[self.rel_paths[idx]].astype(np.float32)
        mspec = getattr(self, "mask_by_dataset", None)      # TUỲ CHỌN (chỉ tập TRAIN): che thuộc tính theo dataset (valid=0)
        if mspec is not None and self.dataset_names[idx] in mspec:
            valid_mask = valid_mask * mspec[self.dataset_names[idx]]
        return {
            "image": image,
            "target": target,          # (40,) float32, 0/1
            "valid_mask": valid_mask,  # (40,) float32, 0 nếu invalid (hiện tại luôn 1 trên train/val)
            "dataset": self.dataset_names[idx],
        }
