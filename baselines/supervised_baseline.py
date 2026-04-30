#!/usr/bin/env python3
"""
Noisy fully-supervised I3D+MLP baseline for wrong-way driving detection.

Training: simple BCE on mean segment score per video. No MIL ranking, no top-k.
The entire positive video is treated as positive; entire negative video as negative.
This establishes the upper bound of what is achievable when weak supervision is
replaced with a (noisy) fully-supervised assumption.

Uses the SAME Model class, feature files, and train/val/test split as DA-RTFM.
Uses ONLY RGB features (no flow) to represent the simplest supervised setting.
Evaluation: calibrate_thresholds.py protocol (val -> threshold -> frozen test eval).

Usage:
    python baselines/supervised_baseline.py \
        --train-list list/splits/train_grouped.list \
        --val-list list/splits/val_base.list \
        --test-list list/splits/test_base.list \
        --seed 42 --num-segments 32 --precision-floor 0.9 \
        --output-json results/supervised_baseline_seed42.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from dataset import Dataset  # noqa: E402
from model import Model, weight_init  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_threshold(labels: np.ndarray, scores: np.ndarray, precision_floor: float) -> float:
    """Select threshold maximizing recall subject to precision >= precision_floor."""
    uniq = np.unique(scores)
    thr_list = np.concatenate(([uniq.min() - 1e-6], uniq, [uniq.max() + 1e-6]))
    best_thr = 0.5
    best = None
    for thr in thr_list:
        preds = (scores >= thr).astype(int)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        if prec < precision_floor:
            continue
        cand = (rec, prec, f1, -float(thr))
        if best is None or cand > best:
            best = cand
            best_thr = float(thr)
    return best_thr


def select_threshold_youdens_j(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select threshold maximising Youden's J = TPR + TNR - 1.

    Always returns a valid threshold.  Handles edge cases (single class,
    constant scores) by falling back to 0.5.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"[WARN] Youden's J: only one class in val — returning 0.5")
        return 0.5
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        print("[WARN] Youden's J: all scores identical — returning 0.5")
        return 0.5
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    preds_mat = (scores[np.newaxis, :] >= unique_scores[:, np.newaxis]).astype(np.int32)
    tp = np.sum(preds_mat[:, labels == 1], axis=1).astype(np.float64)
    tn = n_neg - np.sum(preds_mat[:, labels == 0], axis=1).astype(np.float64)
    tpr = tp / n_pos if n_pos > 0 else np.zeros(len(unique_scores))
    tnr = tn / n_neg if n_neg > 0 else np.zeros(len(unique_scores))
    j = tpr + tnr - 1.0
    return float(unique_scores[int(np.argmax(j))])


class SupervisedArgs:
    """Minimal args namespace compatible with Dataset class."""
    def __init__(
        self,
        rgb_list: str,
        test_rgb_list: str,
        dataset: str = "wrongway-dataset",
        modality: str = "RGB",
        use_flow: bool = False,
        flow_suffix: str = "_flow",
        flow_dim: int = 4,
        split_by_label: bool = True,
    ):
        self.rgb_list = rgb_list
        self.test_rgb_list = test_rgb_list
        self.dataset = dataset
        self.modality = modality
        self.use_flow = use_flow
        self.flow_suffix = flow_suffix
        self.flow_dim = flow_dim
        self.split_by_label = split_by_label


def compute_val_scores(model: Model, loader: DataLoader, device: torch.device):
    """Compute video-level scores and labels using model.infer()."""
    model.eval()
    assert not model.training, "model must be in eval mode"
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for feats, lbl in loader:
            feats = feats.to(device)
            lbl = lbl.to(device)
            video_score, _, _ = model.infer(feats)
            all_scores.append(video_score.detach().cpu().numpy())
            all_labels.append(lbl.detach().cpu().numpy())
    scores = np.concatenate(all_scores, axis=0).reshape(-1).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int32)
    return labels, scores


def train_supervised(
    model: Model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch with supervised BCE on mean segment score per video."""
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    batches = 0
    for feats, labels in train_loader:
        feats = feats.to(device)       # (B, T, F)
        labels = labels.to(device)     # (B,)

        # Use model.infer() which handles the MLP forward pass properly.
        # We use the video_score which is mean of top-k segment scores.
        # For purely supervised baseline, we simply take mean of ALL segments.
        # Replicate the MLP forward pass manually:
        out = feats.permute(0, 2, 1)  # (B, F, T)
        out = model.Aggregate(out)    # (B, T, 512)
        out = model.drop_out(out)
        out = model.relu(model.fc1(out))
        out = model.drop_out(out)
        out = model.relu(model.fc2(out))
        out = model.drop_out(out)
        seg_scores = torch.sigmoid(model.fc3(out)).squeeze(-1)  # (B, T)
        mean_score = seg_scores.mean(dim=1)  # (B,)

        loss = criterion(mean_score, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised I3D+MLP baseline for wrong-way driving detection"
    )
    parser.add_argument("--train-list", required=True, help="Training manifest (path label per line)")
    parser.add_argument("--val-list", required=True, help="Validation manifest")
    parser.add_argument("--test-list", required=True, help="Test manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-root", type=str, default="runs_supervised")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write results JSON",
    )
    parser.add_argument("--precision-floor", type=float, default=0.9)
    parser.add_argument("--feature-size", type=int, default=1024)
    parser.add_argument("--threshold-policy", type=str, default="precision_floor",
                        choices=["precision_floor", "youdens_j"],
                        help="Threshold selection policy on val set")
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = f"results/supervised_baseline_seed{args.seed}.json"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Seed: {args.seed}")

    # Dataset args for training (split_by_label=True -> Dataset filters by class)
    train_args_n = SupervisedArgs(
        rgb_list=args.train_list, test_rgb_list=args.test_list, split_by_label=True
    )
    train_args_a = SupervisedArgs(
        rgb_list=args.train_list, test_rgb_list=args.test_list, split_by_label=True
    )
    # Load full training set (both classes) by using test_mode=True
    train_all_args = SupervisedArgs(
        rgb_list=args.train_list, test_rgb_list=args.train_list, split_by_label=False
    )
    train_set = Dataset(train_all_args, is_normal=True, test_mode=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    # Validation set
    val_args = SupervisedArgs(rgb_list=args.val_list, test_rgb_list=args.val_list, split_by_label=False)
    val_set = Dataset(val_args, is_normal=True, test_mode=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # Test set
    test_args = SupervisedArgs(rgb_list=args.test_list, test_rgb_list=args.test_list, split_by_label=False)
    test_set = Dataset(test_args, is_normal=True, test_mode=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # Model — RGB only, no flow
    model = Model(
        n_features=args.feature_size,
        batch_size=args.batch_size,
        num_segments=args.num_segments,
        topk_ratio=0.25,
    ).to(device)
    model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)

    # Output directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{timestamp}_supervised_seed{args.seed}"
    ckpt_dir = run_dir / "ckpt"
    metrics_dir = run_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    best_epoch = -1

    print(f"\nTraining supervised baseline for {args.max_epoch} epochs...")
    for epoch in range(1, args.max_epoch + 1):
        avg_loss = train_supervised(model, train_loader, optimizer, device, epoch)

        # Validate
        val_labels, val_scores = compute_val_scores(model, val_loader, device)
        val_auc = (
            float(roc_auc_score(val_labels, val_scores))
            if len(set(val_labels.tolist())) > 1
            else float("nan")
        )
        val_auc_pr = (
            float(average_precision_score(val_labels, val_scores))
            if len(set(val_labels.tolist())) > 1
            else float("nan")
        )
        print(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val AUC-ROC: {val_auc:.4f} | Val AUC-PR: {val_auc_pr:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "supervised_best.pkl")
            (metrics_dir / "best_auc.txt").write_text(
                f"epoch={epoch}\nauc_roc={val_auc}\nauc_pr={val_auc_pr}\n", encoding="utf-8"
            )

    print(f"\nBest val AUC-ROC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Load best checkpoint for threshold selection and final eval
    best_ckpt = ckpt_dir / "supervised_best.pkl"
    model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)

    # Select threshold on val set
    val_labels, val_scores = compute_val_scores(model, val_loader, device)
    if getattr(args, "threshold_policy", "precision_floor") == "youdens_j":
        threshold = select_threshold_youdens_j(val_labels, val_scores)
        print(f"Selected threshold={threshold:.6f} (Youden's J)")
    else:
        threshold = select_threshold(val_labels, val_scores, args.precision_floor)
        print(f"Selected threshold={threshold:.6f} (precision_floor={args.precision_floor})")
    val_auc_roc = (
        float(roc_auc_score(val_labels, val_scores)) if len(set(val_labels.tolist())) > 1 else float("nan")
    )
    val_auc_pr = (
        float(average_precision_score(val_labels, val_scores))
        if len(set(val_labels.tolist())) > 1
        else float("nan")
    )

    # Evaluate on test set
    test_labels, test_scores = compute_val_scores(model, test_loader, device)
    preds = (test_scores >= threshold).astype(int)
    test_auc_roc = (
        float(roc_auc_score(test_labels, test_scores))
        if len(set(test_labels.tolist())) > 1
        else float("nan")
    )
    test_auc_pr = (
        float(average_precision_score(test_labels, test_scores))
        if len(set(test_labels.tolist())) > 1
        else float("nan")
    )
    acc = float(accuracy_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, zero_division=0))
    prec = float(precision_score(test_labels, preds, zero_division=0))
    rec = float(recall_score(test_labels, preds, zero_division=0))
    cm = confusion_matrix(test_labels, preds).tolist()
    rep = classification_report(test_labels, preds, digits=4)

    print(f"\nSupervised Baseline Test Results (threshold={threshold:.6f}):")
    print(f"  Test AUC-ROC: {test_auc_roc:.4f}")
    print(f"  Test AUC-PR : {test_auc_pr:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    result = {
        "method": "supervised_i3d_mlp",
        "seed": args.seed,
        "best_val_epoch": best_epoch,
        "val_auc_roc": val_auc_roc,
        "val_auc_pr": val_auc_pr,
        "threshold": threshold,
        "test_auc_roc": test_auc_roc,
        "test_auc_pr": test_auc_pr,
        "test_accuracy": acc,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "test_confusion_matrix": cm,
        "test_report": rep,
        "run_dir": str(run_dir),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
