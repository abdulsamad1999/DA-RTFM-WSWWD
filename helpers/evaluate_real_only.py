#!/usr/bin/env python3
"""Evaluate DA-RTFM checkpoints on the real-only test set using frozen synthetic thresholds.

Usage:
    python helpers/evaluate_real_only.py \
        --runs-root runs_da_rtfm_v3 \
        --filter rtfm_wd_v3 \
        --real-list list/splits/real_test_v2.list \
        --use-flow \
        --num-segments 32 \
        --output-json results/real_only_eval_v3.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from model import Model


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


def _detect_feat_dim(ckpt_path, use_flow):
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Prefer Aggregate conv layers (shape [out, in_feat, k]) — most reliable indicator.
        for k, v in state.items():
            if "Aggregate" in k and "conv" in k and "weight" in k and v.dim() == 3:
                return int(v.shape[1])
        # Fall back to any 2D weight whose input dim looks like a feature dim (>= 512).
        for k, v in state.items():
            if "weight" in k and v.dim() == 2 and v.shape[1] >= 512:
                return int(v.shape[1])
    except Exception:
        pass
    return 1028 if use_flow else 1024


def build_model(ckpt_path, feat_dim, num_segments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(n_features=feat_dim, batch_size=1, topk_ratio=0.25, num_segments=num_segments)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def infer_scores(model, loader, device):
    all_labels, all_scores = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            out = model(inputs)
            seg_scores = out[6].squeeze(-1)
            video_score = seg_scores.max(dim=1).values
            all_scores.extend(video_score.cpu().numpy().tolist())
            if hasattr(labels, "__iter__"):
                all_labels.extend([int(l) for l in labels])
            else:
                all_labels.append(int(labels))
    return np.array(all_labels, dtype=np.int32), np.array(all_scores, dtype=np.float32)


def eval_run(run_dir, real_list, args):
    ckpts = sorted(run_dir.glob("ckpt/*_best.pkl"))
    if not ckpts:
        print(f"  [SKIP] No checkpoint in {run_dir}")
        return None
    ckpt_path = ckpts[-1]

    cal_path = run_dir / "metrics" / "calibration_report.json"
    if not cal_path.exists():
        print(f"  [SKIP] No calibration_report.json in {run_dir}")
        return None
    with open(cal_path, encoding="utf-8") as f:
        cal = json.load(f)
    threshold = float(cal["validation"]["SelectedRule"]["thr"])
    if threshold > 0.99:
        print(f"  [SKIP] Degenerate threshold={threshold:.6f} in {run_dir}")
        return None
    print(f"  Checkpoint: {ckpt_path.name}  threshold={threshold:.6f}")

    feat_dim = _detect_feat_dim(ckpt_path, args.use_flow)
    print(f"  Detected feat_dim={feat_dim}")

    ds_args = argparse.Namespace(
        rgb_list=real_list,
        test_rgb_list=real_list,
        dataset="wrongway-dataset",
        modality="RGB",
        num_segments=args.num_segments,
        feature_size=feat_dim,
        use_flow=args.use_flow,
        flow_suffix="_flow",
        flow_dim=4,
        is_normal=None,
        split_by_label=True,
    )
    try:
        dataset = Dataset(args=ds_args, is_normal=None, transform=None, test_mode=True)
    except TypeError:
        dataset = Dataset(real_list, args.num_segments, is_normal=None, test_mode=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model, device = build_model(ckpt_path, feat_dim, args.num_segments)
    labels, scores = infer_scores(model, loader, device)

    if len(labels) == 0:
        print(f"  [SKIP] No samples in loader for {run_dir}")
        return None

    preds = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    n  = len(labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / n if n > 0 else 0.0

    prec_ci = wilson_ci(precision, tp + fp)
    rec_ci  = wilson_ci(recall,    tp + fn)
    acc_ci  = wilson_ci(accuracy,  n)

    return {
        "run": run_dir.name, "ckpt": ckpt_path.name, "threshold": threshold,
        "n_samples": n, "n_anomaly": int(np.sum(labels == 1)), "n_normal": int(np.sum(labels == 0)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":    round(precision, 4),
        "precision_ci": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
        "recall":       round(recall,    4),
        "recall_ci":    [round(rec_ci[0], 4),  round(rec_ci[1], 4)],
        "f1":           round(f1,        4),
        "accuracy":     round(accuracy,  4),
        "accuracy_ci":  [round(acc_ci[0], 4),  round(acc_ci[1], 4)],
    }


def aggregate(per_run):
    if not per_run:
        return {}
    agg = {}
    for k in ["precision", "recall", "f1", "accuracy"]:
        vals = [r[k] for r in per_run if k in r]
        agg[k] = {"mean": round(float(np.mean(vals)), 4),
                  "std":  round(float(np.std(vals, ddof=0)), 4),
                  "values": [round(v, 4) for v in vals]}
    tp_sum = sum(r["tp"] for r in per_run)
    fp_sum = sum(r["fp"] for r in per_run)
    fn_sum = sum(r["fn"] for r in per_run)
    tn_sum = sum(r["tn"] for r in per_run)
    n_sum  = sum(r["n_samples"] for r in per_run)
    prec_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    rec_p  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    acc_p  = (tp_sum + tn_sum) / n_sum  if n_sum > 0 else 0.0
    agg["pooled"] = {
        "precision":    round(prec_p, 4),
        "precision_ci": [round(v, 4) for v in wilson_ci(prec_p, tp_sum + fp_sum)],
        "recall":       round(rec_p, 4),
        "recall_ci":    [round(v, 4) for v in wilson_ci(rec_p,  tp_sum + fn_sum)],
        "accuracy":     round(acc_p, 4),
        "accuracy_ci":  [round(v, 4) for v in wilson_ci(acc_p,  n_sum)],
        "n_total_predictions": n_sum,
    }
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on real-only test set")
    parser.add_argument("--runs-root",    required=True)
    parser.add_argument("--filter",       required=True)
    parser.add_argument("--real-list",    required=True)
    parser.add_argument("--use-flow",     action="store_true")
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--output-json",  default="results/real_only_eval.json")
    args = parser.parse_args()

    runs_root = REPO_ROOT / args.runs_root
    if not runs_root.is_dir():
        sys.exit(f"ERROR: runs-root not found: {runs_root}")

    real_list = str(REPO_ROOT / args.real_list)
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir() and args.filter in d.name)
    if not run_dirs:
        sys.exit(f"ERROR: No runs matching {args.filter!r} in {runs_root}")

    print(f"Found {len(run_dirs)} runs matching {args.filter!r}")

    per_run_results = []
    for rd in run_dirs:
        print(f"\n--- {rd.name} ---")
        result = eval_run(rd, real_list, args)
        if result is not None:
            per_run_results.append(result)
            print(f"  P={result['precision']:.4f}  R={result['recall']:.4f}  "
                  f"F1={result['f1']:.4f}  Acc={result['accuracy']:.4f}")

    agg = aggregate(per_run_results)
    output = {
        "config": {"runs_root": args.runs_root, "filter": args.filter,
                   "real_list": args.real_list, "use_flow": args.use_flow,
                   "num_segments": args.num_segments},
        "per_run":   per_run_results,
        "aggregate": agg,
    }

    out_path = REPO_ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    if agg:
        print("\n=== Real-Only Evaluation Summary ===")
        print(f"  Runs evaluated: {len(per_run_results)}")
        for k in ["precision", "recall", "f1", "accuracy"]:
            if k in agg:
                print(f"  {k:12s}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")
        if "pooled" in agg:
            p = agg["pooled"]
            print(f"\n  Pooled (Wilson 95% CI):")
            print(f"  precision: {p['precision']:.4f}  CI={p['precision_ci']}")
            print(f"  recall:    {p['recall']:.4f}  CI={p['recall_ci']}")
            print(f"  accuracy:  {p['accuracy']:.4f}  CI={p['accuracy_ci']}")


if __name__ == "__main__":
    main()