#!/usr/bin/env python3
"""Simple flow direction baseline for wrong-way driving detection.

This baseline assumes that the feature files contain the per-segment optical
flow statistics (u, v, mag, D) appended to the RGB features. The anomaly
score for a video is defined as ``1 - mean(D)`` across all segments, where
``D`` is the direction consistency score (cosine of the difference between
the segment flow angle and the dominant traffic direction). A higher score
indicates greater deviation from the dominant direction, hence higher
likelihood of a wrong-way violation.

The script evaluates this baseline by computing AUC-ROC, AUC-PR, accuracy
and F1 on a provided test split. The validation split is used to select a
threshold that satisfies the precision floor.

Usage example::

    python baselines/flow_direction_baseline.py \
        --test-list list/splits/test_base.list \
        --val-list list/splits/val_base.list \
        --use-flow --num-segments 32 \
        --precision-floor 0.9 \
        --output-json results/flow_direction_baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from feature_utils import load_feature_array  # noqa: E402


def load_list(path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items.append((parts[0], int(float(parts[1]))))
    return items


def compute_score(arr: np.ndarray, num_segments: int) -> float:
    """Compute the baseline anomaly score for a single video.

    The last column of ``arr`` is assumed to contain the direction
    consistency score ``D`` per segment. The score is ``1 - mean(D)``.
    If the number of segments in ``arr`` differs from ``num_segments``,
    the features are pooled to ``num_segments`` segments first.
    """
    T, D = arr.shape
    if T != num_segments:
        r = np.linspace(0, T, num_segments + 1, dtype=int)
        pooled = np.zeros((num_segments, D), dtype=np.float32)
        for i in range(num_segments):
            s, e = r[i], r[i + 1]
            pooled[i] = arr[s:e].mean(axis=0) if s < e else arr[s]
        arr = pooled
    D_vals = arr[:, -1]
    return 1.0 - float(np.mean(D_vals))


def select_threshold(labels: np.ndarray, scores: np.ndarray, precision_floor: float) -> float:
    """Select threshold maximizing recall subject to precision >= precision_floor on val set.

    Returns the best threshold. Falls back to 0.5 if no threshold meets the floor.
    """
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


def score_list(items: List[Tuple[str, int]], num_segments: int, flow_suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and compute scores for a list of items."""
    scores_list = []
    labels_list = []
    for feat_path, label in items:
        arr = load_feature_array(
            feat_path,
            num_segments=num_segments,
            use_flow=True,
            flow_suffix=flow_suffix,
        )
        scores_list.append(compute_score(arr, num_segments))
        labels_list.append(label)
    return np.array(scores_list), np.array(labels_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow direction baseline for wrong-way detection")
    parser.add_argument("--test-list", required=True, help="Path to test list file (base scenes only)")
    parser.add_argument(
        "--val-list", type=str, default=None, help="Validation list for threshold selection (required for best results)"
    )
    parser.add_argument("--num-segments", type=int, default=32, help="Number of segments per video")
    parser.add_argument("--precision-floor", type=float, default=0.9, help="Precision floor for threshold selection")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="Default threshold if no validation set")
    parser.add_argument("--flow-suffix", type=str, default="_flow", help="Suffix used for flow sidecar files")
    parser.add_argument("--flow-dim", type=int, default=4, help="Number of flow channels (informational)")
    parser.add_argument(
        "--use-flow",
        action="store_true",
        help="(Ignored; flow is always used by this baseline. Accepted for CLI compatibility.)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON results (default: results/flow_direction_baseline.json)",
    )
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = "results/flow_direction_baseline.json"

    # Load and score test set
    test_items = load_list(args.test_list)
    if not test_items:
        print("ERROR: Test list is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"Scoring {len(test_items)} test videos...")
    test_scores, test_labels = score_list(test_items, args.num_segments, args.flow_suffix)

    # Select threshold on val set
    threshold = args.default_threshold
    val_auc_roc = float("nan")
    val_auc_pr = float("nan")

    if args.val_list:
        val_items = load_list(args.val_list)
        if not val_items:
            print("WARNING: Validation list is empty; using default threshold.", file=sys.stderr)
        else:
            print(f"Scoring {len(val_items)} validation videos for threshold selection...")
            val_scores, val_labels = score_list(val_items, args.num_segments, args.flow_suffix)
            threshold = select_threshold(val_labels, val_scores, args.precision_floor)
            print(
                f"Selected threshold={threshold:.6f} from val set (precision_floor={args.precision_floor})"
            )
            val_auc_roc = (
                float(roc_auc_score(val_labels, val_scores))
                if len(set(val_labels.tolist())) > 1
                else float("nan")
            )
            val_auc_pr = (
                float(average_precision_score(val_labels, val_scores))
                if len(set(val_labels.tolist())) > 1
                else float("nan")
            )
    else:
        print(f"No validation list provided; using default threshold={threshold}")

    # Evaluate on test set
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

    print(f"\nFlow-Direction Baseline Results (threshold={threshold:.6f}):")
    print(f"  Test AUC-ROC : {test_auc_roc:.4f}")
    print(f"  Test AUC-PR  : {test_auc_pr:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1      : {f1:.4f}")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall  : {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{rep}")

    result = {
        "method": "flow_direction",
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
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
