# UPAR 2026 Track 1 — Region-aware EVA-02-L CLIP

Official code and weights for our WACV 2027 UPAR Challenge **Track 1** entry.

**Codabench result: Challenge Avg 0.7948** (mA 0.7994, Instance F1 0.7902).

## Layout

```
inference/              complete Codabench submission package
  run.py                load_model() / predict_batch() API
  model_def.py          EVA-02-L CLIP + region-aware head
  model_config.json     224x112, region head, kNN k=80
  thresholds.json       calibrated per-attribute thresholds
  assets/weights.pt     fp16 checkpoint (Git LFS)
  timm/                 vendored timm 1.0.9 for offline inference
train/                  training and threshold-calibration code
  train.py              training entry point
  run_loo.py            3-fold leave-one-dataset-out training
  calibrate_threshold.py
  configs/              best training config
tools/predict_folder.py run inference on a folder and write CSV
Dockerfile
```

## Weights

`inference/assets/weights.pt` is stored with **Git LFS** (about 606 MB).

```bash
git lfs install
git clone https://github.com/Be-Tap-Code/UPAR-Track-1
cd UPAR-Track-1
git lfs pull
```

## Inference

```bash
python -m pip install -r requirements.txt
python tools/predict_folder.py --images /path/to/images --out predictions.csv
```

The helper calls the same `load_model()` and `predict_batch()` used by the
Codabench ingestion program. Pass the whole gallery in one call: the submitted
model applies transductive kNN smoothing over all images in that call. The CSV
contains one row per image and 40 calibrated probabilities.

Direct API:

```python
import sys
sys.path.insert(0, "inference")
import run
run.load_model()
probs = run.predict_batch([{"image_path": "/path/to/image.jpg"}])[0]
```

## Training

Download the official UPAR 2026 Track-1 data and set `data.root`:

```bash
cd train
python train.py \
  --config configs/cfg_lclip_regionmix.yaml \
  --fold loo_Market1501 \
  --tag best \
  --set data.root=/data/UPAR-Challenge-2027/data
```

The submitted checkpoint is `runs/best_loo_Market1501.pt` (trained on PA100k +
PETA, validated on Market1501). The released `inference/thresholds.json` is
the threshold file used for the leaderboard entry. Use `python train.py --help`
for overrides; the code requires `timm==1.0.9`.

## Docker

```bash
docker build -t upar-track1 .
docker run --gpus all -v /path/to/images:/images upar-track1 \
  python tools/predict_folder.py --images /images --out /workspace/predictions.csv
```

The Docker image uses `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`, matching
the challenge runtime. Training data can be mounted at `/data`.
