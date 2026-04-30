# DA-RTFM: Direction-Consistency Driven Weakly Supervised Wrong-Way Driving Detection

DA-RTFM is a weakly supervised framework for detecting wrong-way driving
in traffic surveillance video using only video-level labels.
It achieves ROC-AUC 0.966+-0.003 on a hybrid synthetic/real benchmark.

**Paper:** [Link to paper when published]
**Dataset:** [Your SharePoint link]
**Code:** This repository

---

## Results

| Metric | Value (mean +- std, 3 seeds) |
|--------|------------------------------|
| ROC-AUC | 0.966 +- 0.003 |
| PR-AUC  | 0.971 +- 0.005 |
| F1      | 0.846 +- 0.015 |
| Precision | 0.975 +- 0.043 |
| Recall  | 0.748 +- 0.018 |
| Latency | 3.06 ms/video (Quadro P1000) |

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch 2.2.1+cu118, CUDA 11.8.

---

## Dataset

Download from: [SharePoint link]

Expected structure after download:
```
dataset/
  training_I3D/Normal/    # *.mp4 training normal clips
  training_I3D/Anomaly/   # *.mp4 training wrong-way clips
  testing_I3D/Normal/     # *.mp4 test normal clips
  testing_I3D/Anomaly/    # *.mp4 test wrong-way clips
```

All clips: 30 FPS, 640x480, 5-12 seconds, GTA5 synthetic + real Lahore footage.

---

## Step 1: Feature Extraction

```bash
# Training features (adaptive temporal resampling, with optical flow)
python helpers/extract_features.py \
  --video-root dataset/training_I3D \
  --output-root features/training_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --save-theta results/theta_ref.json

# Test features
python helpers/extract_features.py \
  --video-root dataset/testing_I3D \
  --output-root features/testing_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --theta-ref results/theta_ref.json
```

Key arguments:
- `--resample-mode adaptive`: divides actual frames into T equal bins (no zero-padding)
- `--use-flow`: appends 4-D optical flow descriptor [u, v, mag, D_t] to I3D features
- `--save-theta`: saves dominant traffic direction theta_ref for direction loss

---

## Step 2: Create Dataset Splits

```bash
# Generate manifests pointing to extracted features
python helpers/prepare_manifests.py generate \
  --video-root dataset/training_I3D \
  --output list/splits/mytrain.list \
  --feature-root features/training_I3D

python helpers/prepare_manifests.py generate \
  --video-root dataset/testing_I3D \
  --output list/splits/mytest.list \
  --feature-root features/testing_I3D

# Grouped scene split (prevents augmentation-variant leakage)
python helpers/split_dataset.py \
  --input list/splits/mytrain.list \
  --output-dir list/splits \
  --val-ratio 0.2 \
  --seed 123 \
  --group-by-base-scene \
  --train-name train_grouped_v3.list \
  --val-name val_grouped_v3.list

# Base-only val and test (no augmented variants in evaluation)
python helpers/create_base_splits.py \
  --val-input list/splits/val_grouped_v3.list \
  --val-output list/splits/val_base_v3.list \
  --test-input list/splits/mytest.list \
  --test-output list/splits/test_base_v3.list
```

---

## Step 3: Training (3 seeds)

```bash
bash scripts/run_da_rtfm.sh
```

Windows:
```
scripts\run_da_rtfm.bat
```

Or manually for one seed:
```bash
python main.py \
  --rgb-list list/splits/train_grouped_v3.list \
  --val-rgb-list list/splits/val_base_v3.list \
  --test-rgb-list list/splits/test_base_v3.list \
  --model-name rtfm_wd \
  --dataset wrongway-dataset \
  --num-segments 32 \
  --topk-ratio 0.25 \
  --use-flow \
  --workers 0 \
  --lambda-temp 1e-4 \
  --temp-on-all \
  --lambda-dir 1e-2 \
  --dir-margin 0.0 \
  --label-smooth-eps 0.1 \
  --seed 42 \
  --output-root runs_da_rtfm
```

Key training arguments:

| Argument | Value | Description |
|----------|-------|-------------|
| `--label-smooth-eps` | 0.1 | Prevents sigmoid saturation from magnitude ranking |
| `--lambda-temp` | 1e-4 | Temporal feature consistency weight |
| `--temp-on-all` | flag | Apply temporal loss to all videos |
| `--lambda-dir` | 1e-2 | Direction-opposition penalty weight |
| `--topk-ratio` | 0.25 | Top-k fraction for magnitude ranking (k=8 for T=32) |

---

## Step 4: Calibration and Evaluation

After training completes, run for each seed:
```bash
python calibrate_thresholds.py \
  --model-path runs_da_rtfm/<run_dir>/ckpt/rtfm_wd_best.pkl \
  --rgb-list list/splits/train_grouped_v3.list \
  --val-rgb-list list/splits/val_base_v3.list \
  --test-rgb-list list/splits/test_base_v3.list \
  --dataset wrongway-dataset \
  --num-segments 32 --topk-ratio 0.25 \
  --use-flow --workers 0 \
  --precision-floor 0.85 \
  --alert-rule video_score \
  --output-report runs_da_rtfm/<run_dir>/metrics/calibration_report.json
```

Frozen test evaluation with calibrated threshold:
```bash
THRESHOLD=$(python -c "import json; \
  d=json.load(open('runs_da_rtfm/<run_dir>/metrics/calibration_report.json')); \
  print(d['validation']['SelectedRule']['thr'])")

python helpers/evaluate_fixed_threshold.py \
  --model-path runs_da_rtfm/<run_dir>/ckpt/rtfm_wd_best.pkl \
  --rgb-list list/splits/train_grouped_v3.list \
  --test-rgb-list list/splits/test_base_v3.list \
  --dataset wrongway-dataset \
  --num-segments 32 --topk-ratio 0.25 \
  --use-flow --workers 0 \
  --threshold $THRESHOLD \
  --split-name test \
  --report-json runs_da_rtfm/<run_dir>/reports/frozen_test_report.json
```

---

## Step 5: Aggregate Multi-Seed Results

```bash
python helpers/compute_run_stats.py \
  --root runs_da_rtfm \
  --filter rtfm_wd
```

---

## Step 6: Run Baselines

```bash
bash scripts/run_baselines.sh
```

Includes: flow-direction baseline, logistic regression, supervised I3D-MLP,
YOLO+tracking (requires `pip install ultralytics`).

---

## Ablation Study

```bash
bash scripts/run_ablation.sh
```

Conditions A (RGB only), B (+Flow), C (+Flow+Temporal) x seeds 42/1337/2026.

| Condition | ROC-AUC | F1 | Description |
|-----------|---------|-----|-------------|
| A | 0.975 +- 0.003 | 0.835 +- 0.062 | RGB features only |
| B | 0.964 +- 0.003 | 0.854 +- 0.047 | + optical flow features |
| C | 0.967 +- 0.003 | 0.846 +- 0.015 | + temporal consistency loss |
| D | 0.967 +- 0.003 | 0.846 +- 0.015 | + flow direction loss (full model) |

---

## Real-World Evaluation

Place real-world videos in:
```
dataset/real-only_I3D/Normal/    # real normal clips
dataset/real-only_I3D/Anomaly/   # real wrong-way clips
```

Extract features and evaluate:
```bash
python helpers/extract_features.py \
  --video-root dataset/real-only_I3D \
  --output-root features/real-only_I3D \
  --num-segments 32 --resample-mode adaptive \
  --use-flow --theta-ref results/theta_ref.json

python helpers/prepare_manifests.py generate \
  --video-root dataset/real-only_I3D \
  --output list/splits/real_test_v2.list \
  --feature-root features/real-only_I3D

python helpers/evaluate_real_only.py \
  --runs-root runs_da_rtfm \
  --real-list list/splits/real_test_v2.list \
  --use-flow --num-segments 32 \
  --output-json results/real_only_eval.json
```

---

## Diagnose Flow Signal

```bash
python helpers/diagnose_flow.py \
  --test-list list/splits/test_base_v3.list \
  --flow-suffix _flow --sample 40 \
  --output-json results/flow_diagnostic.json
```

---

## Project Structure

```
DA-RTFM-release/
+-- main.py                        Training entry point
+-- train.py                       Training loop + loss functions
+-- model.py                       DA-RTFM model (Aggregate + FC scorer)
+-- dataset.py                     WrongwayDataset
+-- option.py                      Argument parser
+-- feature_utils.py               Feature loading utilities
+-- calibrate_thresholds.py        Post-training threshold calibration
+-- pytorch_i3d.py                 I3D backbone (InceptionI3d)
+-- helpers/
|   +-- extract_features.py        I3D + flow feature extraction
|   +-- split_dataset.py           Grouped scene train/val splitting
|   +-- prepare_manifests.py       Manifest generation/verification
|   +-- create_base_splits.py      Filter augmented variants from splits
|   +-- evaluate_fixed_threshold.py Frozen-threshold test evaluation
|   +-- evaluate_real_only.py      Real-world threshold-free evaluation
|   +-- compute_run_stats.py       Multi-seed result aggregation
|   +-- diagnose_flow.py           D_t direction signal diagnostic
|   +-- extract_real_frames.py     Paper: real-world frames + score plots
|   +-- generate_paper_figures.py  Paper: comparison + ablation figures
+-- baselines/
|   +-- flow_direction_baseline.py Rule-based D_t classifier
|   +-- logistic_regression_baseline.py LR on mean-pooled features
|   +-- supervised_baseline.py     Supervised I3D-MLP (no MIL)
|   +-- yolo_tracking_baseline.py  YOLOv8 + centroid tracking
+-- scripts/
    +-- run_da_rtfm.sh / .bat      Full DA-RTFM pipeline (3 seeds)
    +-- run_ablation.sh / .bat     Ablation conditions A/B/C (3 seeds each)
    +-- run_baselines.sh / .bat    All baseline methods
```

---

## Citation

If you use this code, please cite the original RTFM paper:

```bibtex
@inproceedings{tian2021weakly,
  title     = {Weakly-supervised Video Anomaly Detection with Robust
               Temporal Feature Magnitude Learning},
  author    = {Tian, Yu and Pang, Guansong and Chen, Yuanhong and
               Singh, Rajvinder and Verjans, Johan W and Carneiro, Gustavo},
  booktitle = {ICCV},
  year      = {2021}
}
```

And our paper when published:

```bibtex
@article{samad2025dartfm,
  title   = {Direction-Consistency Driven Weakly Supervised Detection of
             Wrong-Way Driving in Traffic Video},
  author  = {Samad, Abdul and Farooq, Muhammad and Jabbar, Sohail and
             Bounceur, Ahcene and Ahmad, Awais and Raza, Umar},
  journal = {[Journal name]},
  year    = {2025}
}
```

---

## License

[Add your license here]