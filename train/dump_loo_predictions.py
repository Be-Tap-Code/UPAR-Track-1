"""Chạy 3 fold LOO (dùng checkpoint đã có sẵn từ run_loo.py), gộp probs/targets/
valid_mask của cả 3 dataset-được-giữ-lại thành 3 file .npy, để calibrate_threshold.py
dùng trực tiếp.

Lý do gộp cả 3 fold thay vì hiệu chỉnh threshold riêng từng fold: threshold cần
ổn định qua nhiều domain (xem strategy.md mục 4.4 và câu hỏi về generalization),
không nên học riêng cho từng dataset nguồn.

Cách dùng (sau khi đã chạy `run_loo.py --tag <tag>` và có checkpoint
runs/<tag>_loo_<Dataset>.pt cho cả 3 dataset):

    python dump_loo_predictions.py --config configs/base.yaml --tag convnext_asl
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data.splits import build_loo_folds
from models.build import build_model
from data.attributes import N_ATTR
from train import build_transforms, make_dataset


@torch.no_grad()
def collect_probs(model, val_loader, device):
    model.eval()
    all_probs, all_targets, all_valid = [], [], []
    for batch in val_loader:
        images = batch["image"].to(device)
        logits = model(images)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(batch["target"].numpy())
        all_valid.append(batch["valid_mask"].numpy())
    return (np.concatenate(all_probs, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_valid, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="tag đã dùng khi chạy run_loo.py")
    ap.add_argument("--out_prefix", default="runs/loo")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_tf = build_transforms(cfg)

    probs_all, targets_all, valid_all = [], [], []
    for fold in build_loo_folds():
        ckpt_path = os.path.join(cfg["output_dir"], f"{args.tag}_{fold.name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Không thấy {ckpt_path}. Chạy `python run_loo.py --config {args.config} "
                f"--tag {args.tag}` trước để tạo checkpoint cho cả 3 fold."
            )
        model = build_model(cfg, N_ATTR).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        val_ds = make_dataset(cfg, [fold.val_dataset], transform=val_tf)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                 num_workers=cfg["data"]["num_workers"])
        probs, targets, valid = collect_probs(model, val_loader, device)
        print(f"[{fold.val_dataset}] {probs.shape[0]} ảnh")
        probs_all.append(probs); targets_all.append(targets); valid_all.append(valid)

    probs_all = np.concatenate(probs_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    valid_all = np.concatenate(valid_all, axis=0)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    np.save(f"{args.out_prefix}_probs.npy", probs_all)
    np.save(f"{args.out_prefix}_targets.npy", targets_all)
    np.save(f"{args.out_prefix}_valid.npy", valid_all)
    print(f"Đã ghi {args.out_prefix}_probs.npy / _targets.npy / _valid.npy "
          f"({probs_all.shape[0]} ảnh gộp cả 3 dataset)")


if __name__ == "__main__":
    main()
