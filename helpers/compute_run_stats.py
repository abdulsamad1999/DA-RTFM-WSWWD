#!/usr/bin/env python3
"""Aggregate performance statistics across multiple run directories."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_best_auc_file(path: Path) -> Dict[str, float]:
    metrics = {"validation_auc_roc": float("nan"), "validation_auc_pr": float("nan")}
    if not path.is_file():
        return metrics
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            try:
                parsed = float(value)
            except ValueError:
                continue
            if key == "auc_roc":
                metrics["validation_auc_roc"] = parsed
            elif key == "auc_pr":
                metrics["validation_auc_pr"] = parsed
    return metrics


def parse_json_metrics(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics: Dict[str, float] = {}
    if "validation" in data:
        metrics["validation_auc_roc"] = float(data["validation"].get("AUC_ROC", float("nan")))
        metrics["validation_auc_pr"] = float(data["validation"].get("AUC_PR", float("nan")))
    if "test" in data:
        metrics["test_accuracy"] = float(data["test"].get("Accuracy", float("nan")))
        metrics["test_f1"] = float(data["test"].get("F1", float("nan")))
        metrics["test_precision"] = float(data["test"].get("Precision", float("nan")))
        metrics["test_recall"] = float(data["test"].get("Recall", float("nan")))
    if "AUC_ROC" in data:
        metrics["test_auc_roc"] = float(data.get("AUC_ROC", float("nan")))
        metrics["test_auc_pr"] = float(data.get("AUC_PR", float("nan")))
        metrics["test_accuracy"] = float(data.get("Accuracy", float("nan")))
        metrics["test_f1"] = float(data.get("F1", float("nan")))
        metrics["test_precision"] = float(data.get("Precision", float("nan")))
        metrics["test_recall"] = float(data.get("Recall", float("nan")))
    return metrics


def collect_metrics(root: Path, filter_str: str) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    if not root.is_dir():
        raise NotADirectoryError(f"Invalid root directory: {root}")
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if filter_str and filter_str not in entry.name:
            continue
        metrics = {"run_name": entry.name}
        metrics.update(parse_best_auc_file(entry / "metrics" / "best_auc.txt"))
        metrics.update(parse_json_metrics(entry / "metrics" / "calibration_report.json"))
        metrics.update(parse_json_metrics(entry / "reports" / "official_test_frozen_report.json"))
        if np.isnan(metrics.get("validation_auc_roc", float("nan"))):
            continue
        results.append(metrics)
    return results


def metric_stats(results: List[Dict[str, float]], key: str) -> Dict[str, float]:
    values = np.array([run.get(key, float("nan")) for run in results], dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def print_stats(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No runs found matching the criteria.")
        return
    metric_keys = [
        ("validation_auc_roc", "Validation AUC-ROC"),
        ("validation_auc_pr", "Validation AUC-PR"),
        ("test_auc_roc", "Test AUC-ROC"),
        ("test_auc_pr", "Test AUC-PR"),
        ("test_accuracy", "Test Accuracy"),
        ("test_f1", "Test F1"),
        ("test_precision", "Test Precision"),
        ("test_recall", "Test Recall"),
    ]
    print(f"Aggregated over {len(results)} runs:")
    for key, label in metric_keys:
        stats = metric_stats(results, key)
        if np.isnan(stats["mean"]):
            continue
        print(f"  {label}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run statistics across DA-RTFM experiments")
    parser.add_argument("--root", type=str, default="runs", help="Root directory containing run subfolders")
    parser.add_argument("--filter", type=str, default="", help="Substring to select specific runs")
    args = parser.parse_args()

    root = Path(args.root)
    results = collect_metrics(root, args.filter)
    if args.filter:
        print(f"Filtering runs containing '{args.filter}' in their names")
    print_stats(results)


if __name__ == "__main__":  # pragma: no cover
    main()
