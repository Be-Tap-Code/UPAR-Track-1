#!/usr/bin/env python3
"""Run the submitted Track-1 model on a folder of images.

This calls the same entry points as the Codabench ingestion program:
    inference/run.py: load_model(), predict_batch()
All images are passed in one call because the inference code performs
transductive kNN smoothing over the whole evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "inference"
sys.path.insert(0, str(INFERENCE))

import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder containing images (searched recursively)")
    ap.add_argument("--out", default="predictions.csv", help="output CSV path")
    ap.add_argument("--ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    args = ap.parse_args()

    exts = {e.lower() for e in args.ext}
    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    with open(INFERENCE / "model_config.json", encoding="utf-8") as f:
        names = json.load(f)["attribute_names"]

    run.load_model()
    samples = [{"image_path": str(p)} for p in paths]
    probs = run.predict_batch(samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", *names])
        for p, row in zip(paths, probs):
            writer.writerow([str(p), *[f"{float(x):.6f}" for x in row]])
    print(f"wrote {out} ({len(paths)} images, {len(names)} attributes)")


if __name__ == "__main__":
    main()
