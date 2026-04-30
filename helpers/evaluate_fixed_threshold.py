#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on a held-out split with a fixed threshold."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
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

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from dataset import Dataset  # noqa: E402
from model import Model  # noqa: E402
from option import parser as option_parser  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen DA-RTFM checkpoint with a fixed threshold",
        add_help=False,
    )
    for action in option_parser._actions:
        if action.option_strings and "-h" not in action.option_strings and "--help" not in action.option_strings:
            parser._add_action(action)
    parser.add_argument("--model-path", required=True, help="Path to the frozen checkpoint")
    parser.add_argument("--threshold", required=True, type=float, help="Fixed threshold selected elsewhere")
    parser.add_argument("--report-json", default=None, help="Optional JSON output path")
    parser.add_argument("--metrics-text", default=None, help="Optional text summary output path")
    parser.add_argument("--split-name", default="heldout", help="Label used in saved reports")
    return parser


def compute_scores(model, loader, device):
    scores = []
    labels = []
    model.eval()
    assert not model.training, "model must be in eval mode"
    with torch.no_grad():
        for feats, lbl in loader:
            feats = feats.to(device)
            lbl = lbl.to(device)
            video_score, _, _ = model.infer(feats)
            scores.append(video_score.detach().cpu().numpy())
            labels.append(lbl.detach().cpu().numpy())
    return (
        np.concatenate(labels, axis=0).reshape(-1).astype(np.int32),
        np.concatenate(scores, axis=0).reshape(-1).astype(np.float64),
    )


def ensure_parent(path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if not args.test_rgb_list:
        raise ValueError("Evaluation requires --test-rgb-list to point at the held-out manifest.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_feature_size = args.feature_size + (args.flow_dim if getattr(args, "use_flow", False) else 0)
    model = Model(
        n_features=input_feature_size,
        batch_size=args.batch_size,
        num_segments=args.num_segments,
        topk_ratio=getattr(args, "topk_ratio", 0.25),
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    eval_args = argparse.Namespace(**vars(args))
    eval_set = Dataset(eval_args, is_normal=True, test_mode=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    labels, scores = compute_scores(model, eval_loader, device)
    preds = (scores >= float(args.threshold)).astype(np.int32)

    result = {
        "split": args.split_name,
        "threshold": float(args.threshold),
        "AUC_ROC": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan"),
        "AUC_PR": average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan"),
        "Accuracy": accuracy_score(labels, preds),
        "F1": f1_score(labels, preds, zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "ConfusionMatrix": confusion_matrix(labels, preds).tolist(),
        "Report": classification_report(labels, preds, digits=4),
    }

    print(f"=== {args.split_name} Evaluation ===")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"AUC-ROC: {result['AUC_ROC']:.4f}")
    print(f"AUC-PR:  {result['AUC_PR']:.4f}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"F1:       {result['F1']:.4f}")
    print(f"Precision:{result['Precision']:.4f}")
    print(f"Recall:   {result['Recall']:.4f}")

    if args.report_json:
        ensure_parent(args.report_json)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote JSON report to {args.report_json}")
    if args.metrics_text:
        ensure_parent(args.metrics_text)
        with open(args.metrics_text, "w", encoding="utf-8") as f:
            for key in ("split", "threshold", "AUC_ROC", "AUC_PR", "Accuracy", "F1", "Precision", "Recall"):
                f.write(f"{key}={result[key]}\n")
            f.write(f"ConfusionMatrix={result['ConfusionMatrix']}\n")
        print(f"Wrote text metrics to {args.metrics_text}")


if __name__ == "__main__":  # pragma: no cover
    main()
