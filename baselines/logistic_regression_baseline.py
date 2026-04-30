#!/usr/bin/env python3
"""Logistic regression baseline for wrong-way driving anomaly detection.

Aggregates per-segment features into a fixed-length vector per video (mean pooling),
trains a logistic regression classifier on the training split, selects a threshold
on the validation split under a precision floor, and evaluates on the test split.

Usage::

    python baselines/logistic_regression_baseline.py \
        --train-list list/splits/train_grouped.list \
        --val-list list/splits/val_base.list \
        --test-list list/splits/test_base.list \
        --num-segments 32 --precision-floor 0.9 \
        --output-json results/lr_baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from feature_utils import load_feature_array  # noqa: E402
from option import parser as option_parser  # noqa: E402


def load_list(list_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items.append((parts[0], int(float(parts[1]))))
    return items


def aggregate_feature(path: str, num_segments: int, use_flow: bool, flow_suffix: str) -> np.ndarray:
    """Load a .npy feature file and average across the temporal dimension."""
    arr = load_feature_array(path, num_segments=num_segments, use_flow=use_flow, flow_suffix=flow_suffix)
    return arr.mean(axis=0)


def select_threshold(labels: np.ndarray, scores: np.ndarray, precision_floor: float) -> float:
    """Select threshold maximizing recall subject to precision >= precision_floor on val set."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic regression baseline for wrong-way detection")
    parser.add_argument("--train-list", required=True, help="Path to training list file")
    parser.add_argument("--val-list", type=str, default=None, help="Validation list for threshold selection")
    parser.add_argument("--test-list", required=True, help="Path to testing list file")
    parser.add_argument("--num-segments", type=int, default=32, help="Number of segments per video (default 32)")
    parser.add_argument("--use-flow", action="store_true", help="If set, features include flow channels")
    parser.add_argument("--flow-dim", type=int, default=4, help="Number of flow channels when --use-flow is set")
    parser.add_argument("--flow-suffix", type=str, default="_flow", help="Suffix used for flow sidecar files")
    parser.add_argument("--regularization", type=float, default=1.0, help="Inverse regularization strength (C)")
    parser.add_argument("--precision-floor", type=float, default=0.9, help="Precision floor for threshold selection")
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to save the trained model (.pkl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON results (default: results/lr_baseline.json)",
    )
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = "results/lr_baseline.json"

    feat_dim = option_parser.get_default("feature_size")
    if args.use_flow:
        feat_dim += args.flow_dim

    np.random.seed(args.seed)

    train_items = load_list(args.train_list)
    test_items = load_list(args.test_list)
    if not train_items or not test_items:
        print("ERROR: Training or testing list is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(train_items)} training features (feat_dim={feat_dim})...")
    X_train = np.zeros((len(train_items), feat_dim), dtype=np.float32)
    y_train = np.zeros((len(train_items),), dtype=np.int32)
    for i, (feat_path, label) in enumerate(train_items):
        X_train[i] = aggregate_feature(feat_path, args.num_segments, args.use_flow, args.flow_suffix)
        y_train[i] = label

    print(f"Loading {len(test_items)} test features...")
    X_test = np.zeros((len(test_items), feat_dim), dtype=np.float32)
    y_test = np.zeros((len(test_items),), dtype=np.int32)
    for i, (feat_path, label) in enumerate(test_items):
        X_test[i] = aggregate_feature(feat_path, args.num_segments, args.use_flow, args.flow_suffix)
        y_test[i] = label

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    print("Training logistic regression...")
    clf = LogisticRegression(C=args.regularization, max_iter=1000, random_state=args.seed)
    clf.fit(X_train, y_train)

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "scaler": scaler}, args.save_model)
        print(f"Saved model to {args.save_model}")

    # Select threshold on val set if provided
    threshold = 0.5
    val_auc_roc = float("nan")
    val_auc_pr = float("nan")

    if args.val_list:
        val_items = load_list(args.val_list)
        if not val_items:
            print("WARNING: Validation list is empty; using default threshold=0.5.", file=sys.stderr)
        else:
            print(f"Loading {len(val_items)} validation features for threshold selection...")
            X_val = np.zeros((len(val_items), feat_dim), dtype=np.float32)
            y_val = np.zeros((len(val_items),), dtype=np.int32)
            for i, (feat_path, label) in enumerate(val_items):
                X_val[i] = aggregate_feature(feat_path, args.num_segments, args.use_flow, args.flow_suffix)
                y_val[i] = label
            X_val = scaler.transform(X_val)
            val_probs = clf.predict_proba(X_val)[:, 1]
            threshold = select_threshold(y_val, val_probs, args.precision_floor)
            print(f"Selected threshold={threshold:.6f} (precision_floor={args.precision_floor})")
            val_auc_roc = (
                float(roc_auc_score(y_val, val_probs)) if len(set(y_val.tolist())) > 1 else float("nan")
            )
            val_auc_pr = (
                float(average_precision_score(y_val, val_probs))
                if len(set(y_val.tolist())) > 1
                else float("nan")
            )
    else:
        print("No val list provided; using default threshold=0.5")

    # Evaluate on test
    prob_test = clf.predict_proba(X_test)[:, 1]
    preds = (prob_test >= threshold).astype(int)
    test_auc_roc = (
        float(roc_auc_score(y_test, prob_test)) if len(set(y_test.tolist())) > 1 else float("nan")
    )
    test_auc_pr = (
        float(average_precision_score(y_test, prob_test))
        if len(set(y_test.tolist())) > 1
        else float("nan")
    )
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()
    rep = classification_report(y_test, preds, digits=4)

    print(f"\nLogistic Regression Baseline Results (threshold={threshold:.6f}):")
    print(f"  Test AUC-ROC : {test_auc_roc:.4f}")
    print(f"  Test AUC-PR  : {test_auc_pr:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1      : {f1:.4f}")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall  : {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{rep}")

    result = {
        "method": "logistic_regression",
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
