"""Chạy đủ 3 fold leave-one-dataset-out, gộp kết quả theo đúng cách ban tổ chức
tổng hợp nhiều dataset (trung bình cộng qua dataset), in bảng so sánh.

Đây là lệnh chính để QUYẾT ĐỊNH có giữ một thay đổi cấu hình hay không:
    python run_loo.py --config configs/base.yaml --tag baseline_convnext
So sánh Challenge_Avg và std giữa các lần chạy có tag khác nhau trong
runs/loo_results.jsonl trước khi merge thay đổi vào base.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from data.splits import build_loo_folds
from eval.metrics import aggregate_datasets
from train import make_dataset, build_transforms, train_one_fold, run_validation
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_batches", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    _, val_tf = build_transforms(cfg)
    per_dataset_results = []
    for fold in build_loo_folds():
        result = train_one_fold(cfg, fold, device=device, epochs=args.epochs, limit_batches=args.limit_batches)
        model = result["model"]
        # strict=False: state_dict đã bị strip domain_classifier (chỉ cần lúc
        # train DANN, xem train.py/strip_domain_classifier) — model gốc ở đây
        # vẫn còn domain_classifier (nếu DANN bật) nên sẽ báo "missing keys" ở
        # strict=True dù không ảnh hưởng gì tới forward() dùng để eval bên dưới.
        model.load_state_dict(result["state_dict"], strict=False)

        val_ds = make_dataset(cfg, [fold.val_dataset], transform=val_tf)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                 num_workers=cfg["data"]["num_workers"])
        metrics = run_validation(model, val_loader, device, limit_batches=args.limit_batches)
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        print(f"[{fold.val_dataset}] {metrics_clean}")
        per_dataset_results.append(metrics)

        ckpt_path = os.path.join(cfg["output_dir"], f"{args.tag}_{fold.name}.pt")
        torch.save(result["state_dict"], ckpt_path)

    agg = aggregate_datasets(per_dataset_results)
    print("\n=== LOO aggregate (trung bình + std qua 3 dataset, đúng cách chấm) ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    log_path = os.path.join(cfg["output_dir"], "loo_results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"tag": args.tag, "config": args.config, "aggregate": agg}) + "\n")
    print(f"\nĐã ghi vào {log_path}")


if __name__ == "__main__":
    main()
