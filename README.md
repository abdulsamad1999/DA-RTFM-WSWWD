# DA-RTFM: Direction-Consistency Driven Weakly Supervised Wrong-Way Driving Detection

**DA-RTFM** (Direction-Aware Robust Temporal Feature Magnification) is a weakly supervised framework for detecting wrong-way driving in traffic surveillance video using only video-level binary labels — no frame-level annotations, no camera calibration, and no tracking pipeline required.

**Paper:** To be updated on publication  
**Dataset, Features & Weights:** [SharePoint](https://pern-my.sharepoint.com/personal/abdul_1617851_talmeez_pk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabdul%5F1617851%5Ftalmeez%5Fpk%2FDocuments%2FWSWWD&ga=1)  
**Code:** [https://github.com/abdulsamad1999/DA-RTFM-WSWWD](https://github.com/abdulsamad1999/DA-RTFM-WSWWD)

---

## Results

**Synthetic test set** (n=65 base scenes, mean ± std over 3 seeds):

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.966 ± 0.003 |
| PR-AUC | 0.971 ± 0.005 |
| F1 | 0.846 ± 0.015 |
| Precision | 0.975 ± 0.043 |
| Recall | 0.748 ± 0.018 |
| Inference latency | 3.06 ms/video (Quadro P1000, batch=1) |
| Throughput | 697 videos/sec |
| Peak GPU memory | 77.8 MB |

**Real-world zero-shot transfer** (n=88 real clips): ROC-AUC 0.560, Recall 0.924.  
High sensitivity transfers from synthetic training; precision is limited by I3D feature domain shift (GTA5 → real footage). See the paper for a detailed analysis.

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+  
- PyTorch 2.2.1 with CUDA 11.8  
- OpenCV 4.8+  
- scikit-learn 1.3+

---

## SharePoint Downloads

All resources needed to reproduce the published results are available at the SharePoint link above. The SharePoint contains:

```
WSWWD/
  dataset/          ← raw .mp4 video files (4 subfolders)
  features_v2/      ← pre-extracted .npy features — use this to skip extraction (~6 hours)
  results/          ← pre-computed calibration reports and evaluation outputs
  rgb_imagenet.pt   ← I3D backbone weights (48.5 MB, required for feature extraction)
  yolov8n.pt        ← YOLOv8n weights (6.25 MB, required only for the YOLO+Tracking baseline)
```

### Option A — Pre-extracted features (recommended)

Download `features_v2/` from SharePoint and place it in the project root:

```
DA-RTFM-WSWWD/
  features_v2/
    training_I3D/Normal/       ← .npy files
    training_I3D/Anomaly/
    validation_I3D/Normal/
    validation_I3D/Anomaly/
    testing_I3D/Normal/
    testing_I3D/Anomaly/
    real-only_I3D/Normal/
    real-only_I3D/Anomaly/
```

> **Important:** The validation split (`val_base_v3.list`) draws clips from **both** `features_v2/training_I3D/` and `features_v2/validation_I3D/` because the combined-pool grouped split redistributed scenes across both source folders. Download the complete `features_v2/` folder — not individual subfolders — to ensure all validation paths resolve correctly.

Feature shape: `(32, 1028)` per video — 32 adaptive temporal segments × 1028-D (1024 I3D-RGB + 4 optical-flow: u, v, magnitude, direction cosine D_t).

### Option B — Extract features from raw videos

Download `dataset/` and `rgb_imagenet.pt` from SharePoint. Place `rgb_imagenet.pt` in the project root, then follow the Feature Extraction section below.

### Option C — Use pretrained checkpoints (skip training)

Download pretrained `.pkl` checkpoints from SharePoint (under `results/` or a dedicated `checkpoints/` folder) and place them as:

```
DA-RTFM-WSWWD/
  checkpoints/
    seed42_rtfm_wd_best.pkl
    seed1337_rtfm_wd_best.pkl
    seed2026_rtfm_wd_best.pkl
```

Then run calibration and evaluation directly (Steps 3 and 4 below), skipping training entirely. Expected results per seed: seed 42 — ROC-AUC 0.9665, F1 0.8621, threshold 0.833.

---

## Pre-provided Split Files

The train/val/test split files used to produce the published results are included in this repository and require no modification:

```
list/splits/
  train_grouped_v3.list   ← 876 training entries (265 base scenes × 3 augmentations)
  val_base_v3.list        ← 73 validation entries (base scenes only, no augmentation)
  test_base_v3.list       ← 65 synthetic test entries (base scenes only)
  real_test_v2.list       ← 88 real-world test entries
```

There is no need to re-run split generation.

---

## Step 1: Feature Extraction (skip if using Option A)

Download `rgb_imagenet.pt` from SharePoint and place it in the project root, then run:

```bash
# Training features — saves reference flow direction theta_ref
python helpers/extract_features.py \
  --video-root dataset/training_I3D \
  --output-root features_v2/training_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --save-theta results/theta_ref.json

# Validation features
python helpers/extract_features.py \
  --video-root dataset/validation_I3D \
  --output-root features_v2/validation_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --theta-ref results/theta_ref.json

# Synthetic test features
python helpers/extract_features.py \
  --video-root dataset/testing_I3D \
  --output-root features_v2/testing_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --theta-ref results/theta_ref.json

# Real-world test features
python helpers/extract_features.py \
  --video-root dataset/real-only_I3D \
  --output-root features_v2/real-only_I3D \
  --num-segments 32 \
  --resample-mode adaptive \
  --use-flow \
  --theta-ref results/theta_ref.json
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--resample-mode adaptive` | Divides actual frames into T equal-duration bins with no zero-padding |
| `--use-flow` | Appends 4-D flow descriptor [u, v, magnitude, D_t] to the 1024-D I3D features |
| `--save-theta` | Estimates and saves the dominant traffic direction θ_ref from training normals |
| `--theta-ref` | Loads θ_ref saved from training for consistent direction encoding |

---

## Step 2: Training (3 seeds)

```bash
# Linux/macOS
bash scripts/run_da_rtfm.sh

# Windows
scripts\run_da_rtfm.bat
```

To run a single seed manually:

```bash
python main.py \
  --rgb-list        list/splits/train_grouped_v3.list \
  --val-rgb-list    list/splits/val_base_v3.list \
  --test-rgb-list   list/splits/test_base_v3.list \
  --model-name      rtfm_wd \
  --dataset         wrongway-dataset \
  --num-segments    32 \
  --topk-ratio      0.25 \
  --use-flow \
  --workers         0 \
  --lambda-temp     1e-4 \
  --temp-on-all \
  --lambda-dir      1e-2 \
  --dir-margin      0.0 \
  --label-smooth-eps 0.1 \
  --seed            42 \
  --output-root     runs_da_rtfm
```

Key training arguments:

| Argument | Value | Purpose |
|----------|-------|---------|
| `--label-smooth-eps` | 0.1 | Prevents sigmoid saturation caused by magnitude-ranking pressure; required for reliable threshold calibration |
| `--lambda-temp` | 1e-4 | Temporal feature consistency regulariser weight; reduces seed-to-seed F1 variance by 3× |
| `--temp-on-all` | flag | Apply temporal consistency loss to both normal and anomalous videos |
| `--lambda-dir` | 1e-2 | Direction-opposition penalty weight (applied to normal videos only) |
| `--topk-ratio` | 0.25 | Top-k fraction for magnitude ranking — k=8 for T=32 segments |

---

## Step 3: Threshold Calibration

After training, calibrate the detection threshold on the validation set for each seed:

```bash
python calibrate_thresholds.py \
  --model-path   runs_da_rtfm/<run_dir>/ckpt/rtfm_wd_best.pkl \
  --rgb-list     list/splits/train_grouped_v3.list \
  --val-rgb-list list/splits/val_base_v3.list \
  --test-rgb-list list/splits/test_base_v3.list \
  --dataset      wrongway-dataset \
  --num-segments 32 --topk-ratio 0.25 \
  --use-flow --workers 0 \
  --precision-floor 0.85 \
  --alert-rule   video_score \
  --output-report runs_da_rtfm/<run_dir>/metrics/calibration_report.json
```

---

## Step 4: Frozen Test Evaluation

```bash
# Extract the calibrated threshold from the calibration report
THRESHOLD=$(python -c "
import json
d = json.load(open('runs_da_rtfm/<run_dir>/metrics/calibration_report.json'))
print(d['validation']['SelectedRule']['thr'])
")

# Evaluate on the frozen synthetic test set
python helpers/evaluate_fixed_threshold.py \
  --model-path    runs_da_rtfm/<run_dir>/ckpt/rtfm_wd_best.pkl \
  --rgb-list      list/splits/train_grouped_v3.list \
  --test-rgb-list list/splits/test_base_v3.list \
  --dataset       wrongway-dataset \
  --num-segments  32 --topk-ratio 0.25 \
  --use-flow --workers 0 \
  --threshold     $THRESHOLD \
  --split-name    test \
  --report-json   runs_da_rtfm/<run_dir>/reports/frozen_test_report.json
```

---

## Step 5: Aggregate Multi-Seed Results

```bash
python helpers/compute_run_stats.py \
  --root   runs_da_rtfm \
  --filter rtfm_wd
```

---

## Step 6: Real-World Evaluation

```bash
python helpers/evaluate_real_only.py \
  --runs-root  runs_da_rtfm \
  --filter     rtfm_wd \
  --real-list  list/splits/real_test_v2.list \
  --use-flow   --num-segments 32 \
  --output-json results/real_only_eval.json
```

---

## Ablation Study

```bash
bash scripts/run_ablation.sh    # Linux/macOS
scripts\run_ablation.bat        # Windows
```

| Condition | ROC-AUC | PR-AUC | F1 | Key finding |
|-----------|---------|--------|-----|-------------|
| A — RGB only | 0.975 ± 0.003 | 0.979 ± 0.003 | 0.835 ± 0.062 | Strongest ranking; wide F1 variance |
| B — +Flow | 0.964 ± 0.003 | 0.971 ± 0.001 | 0.854 ± 0.047 | Precision = 1.000 ± 0.000; zero false positives |
| C — +Temporal | 0.967 ± 0.003 | 0.971 ± 0.005 | 0.846 ± 0.015 | F1 variance reduced 3× vs B |
| D — Full DA-RTFM | 0.967 ± 0.003 | 0.971 ± 0.005 | 0.846 ± 0.015 | C = D; direction loss inactive on moving-camera footage |

The direction loss ($\mathcal{L}_\text{dir}$) is currently inactive because scene-level optical flow on moving-camera video is dominated by camera ego-motion, not individual vehicle direction. Object-level flow (vehicle detection before flow aggregation) is required to activate this component.

---

## Baselines

Download `yolov8n.pt` from SharePoint for the YOLO+Tracking baseline. All other baselines have no extra dependencies.

```bash
bash scripts/run_baselines.sh    # Linux/macOS
scripts\run_baselines.bat        # Windows
```

| Baseline | ROC-AUC | Type |
|----------|---------|------|
| YOLO + Tracking | 0.605 | Task-specific (supervised) |
| Flow + Rules | 0.612 | Heuristic |
| BN-WVAD | 0.626 | Weakly supervised |
| VadCLIP | 0.630 | Weakly supervised |
| ProDISC-VAD | 0.717 | Weakly supervised |
| MIL | 0.827 | Weakly supervised |
| Logistic Regression (I3D) | 0.843 | Linear baseline |
| UMIL | 0.858 | Weakly supervised |
| Supervised I3D-MLP | 0.940 ± 0.017 | Supervised upper bound |
| **DA-RTFM (Ours)** | **0.966 ± 0.003** | **Weakly supervised** |

---

## Flow Diagnostic

To check the direction signal quality on any split:

```bash
python helpers/diagnose_flow.py \
  --test-list   list/splits/test_base_v3.list \
  --flow-suffix _flow \
  --sample      40 \
  --output-json results/flow_diagnostic.json
```

Expected output on the current dataset: D_t mean ≈ 0.88, fewer than 2% negative values. This confirms that scene-level flow on the current footage is dominated by camera/background motion. Object-level flow via vehicle detection is required to activate the direction-opposition component.

---

## Project Structure

```
DA-RTFM-WSWWD/
├── main.py                          Training entry point
├── train.py                         Training loop and RTFM-M + regulariser losses
├── model.py                         DA-RTFM temporal head (dilated conv + attention + MLP)
├── dataset.py                       WrongwayDataset loader
├── option.py                        All CLI argument definitions
├── feature_utils.py                 Feature loading and normalisation utilities
├── calibrate_thresholds.py          Post-training threshold selection (Youden's J)
├── pytorch_i3d.py                   I3D backbone (requires rgb_imagenet.pt)
├── requirements.txt
├── helpers/
│   ├── extract_features.py          I3D + Farnebäck optical flow extraction
│   ├── evaluate_fixed_threshold.py  Frozen-threshold evaluation on test set
│   ├── evaluate_real_only.py        Real-world evaluation with Wilson 95% CI
│   ├── compute_run_stats.py         Multi-seed result aggregation (mean ± std)
│   ├── diagnose_flow.py             D_t direction signal diagnostic
│   ├── create_base_splits.py        Filter augmented variants from split files
│   ├── prepare_manifests.py         Manifest generation from feature folders
│   ├── split_dataset.py             Grouped scene train/val split
│   └── generate_paper_figures.py    Reproduces paper figures from results
├── baselines/
│   ├── flow_direction_baseline.py   Rule-based D_t threshold classifier
│   ├── logistic_regression_baseline.py  LR on mean-pooled I3D features
│   ├── supervised_baseline.py       Noisy fully-supervised I3D-MLP
│   └── yolo_tracking_baseline.py    YOLOv8n + centroid tracking (needs yolov8n.pt)
├── scripts/
│   ├── run_da_rtfm.sh / .bat        Full DA-RTFM training pipeline (3 seeds)
│   ├── run_ablation.sh / .bat       Ablation conditions A/B/C × 3 seeds
│   └── run_baselines.sh / .bat      All baseline methods
└── list/splits/
    ├── train_grouped_v3.list        876 training entries (aug. included)
    ├── val_base_v3.list             73 validation entries (base only)
    ├── test_base_v3.list            65 synthetic test entries (base only)
    └── real_test_v2.list            88 real-world test entries
```

---

## Note on Validation Split

The validation set (`val_base_v3.list`) was constructed by merging the training pool with an independent GTA5 validation set, then applying a grouped 80/20 scene split on the combined pool. As a result, validation feature paths point to **both** `features_v2/training_I3D/` and `features_v2/validation_I3D/` — specifically, 22 of 35 normal val clips and 30 of 38 anomaly val clips are drawn from `features_v2/training_I3D/`. Download the complete `features_v2/` folder to ensure all paths resolve correctly. Scene-level separation between train and validation is guaranteed (zero overlap) despite the shared feature subfolder.

---

## Citation

If you use this work, please cite:

```bibtex
@article{samad2025dartfm,
  title   = {Direction-Consistency Driven Weakly Supervised Detection of
             Wrong-Way Driving in Traffic Video},
  author  = {Samad, Abdul and Farooq, Muhammad and Jabbar, Sohail and
             Bounceur, Ahcene and Ahmad, Awais and Raza, Umar},
  journal = {[Journal name — to be updated on publication]},
  year    = {2025}
}
```

This work builds on RTFM:

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

---

## License

[Add your license here]
