"""
Calibration and test evaluation for wrong‑way driving anomaly detection.

This script separates threshold calibration from final test evaluation. It
performs search for an operating threshold (either on video scores or via
segment‑persistence alerting) using a validation set, and then applies the
selected threshold to the test set. This avoids leaking test labels into
model selection and follows best practices for anomaly detection under weak
supervision.

Usage Example:

    python calibrate_thresholds.py \
        --dataset wrongway-dataset \
        --rgb-list list/mytrain.list \
        --val-rgb-list list/myval.list \
        --test-rgb-list list/mytest.list \
        --use-flow --flow-suffix _flow --flow-dim 4 \
        --topk-ratio 0.25 --num-segments 32 --batch-size 16 \
        --model-path runs/ckpt/best.pkl \
        --precision-floor 0.85 --alert-rule auto --m-grid 3,4,5,6,8 \
        --k-grid-mode strict

The output prints both validation and test metrics at the chosen operating
threshold and writes a JSON report to the specified output directory (if
provided).
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from option import parser as option_parser
from dataset import Dataset
from model import Model
# Local implementations of threshold and alerting utilities. The original
# code imported these from test_safety_alerting.py, which has been
# removed. They are reimplemented here for standalone usage.

def candidate_thresholds(scores):
    """Generate a sorted list of unique candidate thresholds from scores.

    Args:
        scores (np.ndarray): 1D array of anomaly scores.
    Returns:
        np.ndarray of unique thresholds sorted ascending.
    """
    scores = np.asarray(scores).astype(np.float64).flatten()
    uniq = np.unique(scores)
    # Include a small epsilon above and below to allow margin search
    eps = 1e-6
    thr_list = np.concatenate(([uniq.min() - eps], uniq, [uniq.max() + eps]))
    return np.sort(thr_list)


def segment_persistence_alert(seg_scores, threshold: float, k: int, m: int):
    """Apply a segment persistence rule to determine anomaly labels.

    For each video, a sliding window of length `m` frames is used. If
    at least `k` segments within any window exceed the specified
    threshold, the video is flagged as anomalous. Otherwise it is
    classified as normal.

    Args:
        seg_scores (np.ndarray): Array of shape (N, T) containing per-segment scores for N videos.
        threshold (float): Threshold on segment scores.
        k (int): Minimum number of segments within a window to raise alert.
        m (int): Size of the sliding window.

    Returns:
        np.ndarray of shape (N,) with binary predictions (1=anomaly, 0=normal).
    """
    seg_scores = np.asarray(seg_scores)
    n_videos, n_segments = seg_scores.shape
    preds = np.zeros((n_videos,), dtype=np.int32)
    for i in range(n_videos):
        scores = seg_scores[i]
        alerted = False
        for start in range(0, n_segments - m + 1):
            window = scores[start:start + m]
            if np.sum(window >= threshold) >= k:
                alerted = True
                break
        preds[i] = 1 if alerted else 0
    return preds


def metrics(labels, preds):
    """Compute standard classification metrics.

    Args:
        labels (np.ndarray): Ground truth binary labels.
        preds (np.ndarray): Predicted binary labels.
    Returns:
        Tuple (Accuracy, F1, Precision, Recall, ConfusionMatrix, ClassificationReport)
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        classification_report,
    )
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    rep = classification_report(labels, preds, digits=4)
    return acc, f1, prec, rec, cm, rep


def select_video_threshold_max_recall(labels, video_scores, precision_floor: float):
    """Select a video‑level threshold on the validation set.

    Returns (best_threshold, metrics_dict) where metrics_dict contains
    Accuracy, F1, Precision, Recall, ConfusionMatrix, Report.
    """
    thr_list = candidate_thresholds(video_scores)
    best = None
    best_thr = 0.5
    best_pack = None

    for thr in thr_list:
        preds = (video_scores >= float(thr)).astype(np.int32)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        if prec < precision_floor:
            continue
        cand = (rec, prec, f1, -float(thr))
        if best is None or cand > best:
            best = cand
            best_thr = float(thr)
            best_pack = {
                "Accuracy": acc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
                "ConfusionMatrix": cm.tolist(),
                "Report": rep,
            }
    return best_thr, best_pack


def select_threshold_youdens_j(labels, video_scores):
    """Select threshold that maximises Youden's J statistic on the val set.

    J = sensitivity + specificity - 1 = TPR + TNR - 1

    Equivalent to finding the point on the ROC curve farthest from the
    diagonal.  Cannot produce a degenerate threshold; always returns a
    valid operating point.

    Edge cases:
      - All scores identical  → threshold = 0.5
      - Only one class in val → threshold = 0.5 with warning

    Returns:
        (threshold, metrics_dict)  where metrics_dict contains Accuracy,
        F1, Precision, Recall, ConfusionMatrix, Report.
    """
    labels = np.asarray(labels, dtype=np.int32)
    video_scores = np.asarray(video_scores, dtype=np.float64)

    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"[WARN] select_threshold_youdens_j: only one class ({unique_classes}) "
              "in val labels — returning threshold=0.5")
        preds = (video_scores >= 0.5).astype(np.int32)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        return 0.5, {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
                     "ConfusionMatrix": cm.tolist(), "Report": rep}

    unique_scores = np.unique(video_scores)
    if len(unique_scores) == 1:
        print("[WARN] select_threshold_youdens_j: all scores identical — returning threshold=0.5")
        preds = (video_scores >= 0.5).astype(np.int32)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        return 0.5, {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
                     "ConfusionMatrix": cm.tolist(), "Report": rep}

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    # Vectorise over all unique score values as candidate thresholds.
    # Shape: (n_thresholds, n_samples)
    preds_mat = (video_scores[np.newaxis, :] >= unique_scores[:, np.newaxis]).astype(np.int32)

    tp = np.sum(preds_mat[:, labels == 1], axis=1).astype(np.float64)
    fp = np.sum(preds_mat[:, labels == 0], axis=1).astype(np.float64)
    fn = n_pos - tp
    tn = n_neg - fp

    tpr = np.where(n_pos > 0, tp / n_pos, 0.0)   # sensitivity / recall
    tnr = np.where(n_neg > 0, tn / n_neg, 0.0)   # specificity
    j   = tpr + tnr - 1.0

    best_idx = int(np.argmax(j))
    best_thr = float(unique_scores[best_idx])

    preds = preds_mat[best_idx]
    acc, f1, prec, rec, cm, rep = metrics(labels, preds)
    pack = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
            "ConfusionMatrix": cm.tolist(), "Report": rep}
    return best_thr, pack


def select_persistence_threshold_max_recall(labels, seg_scores, k: int, m: int, precision_floor: float):
    thr_list = candidate_thresholds(seg_scores)
    best = None
    best_thr = None
    best_pack = None

    for thr in thr_list:
        preds = segment_persistence_alert(seg_scores, float(thr), k=k, m=m)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        if prec < precision_floor:
            continue
        cand = (rec, prec, f1, -float(thr))
        if best is None or cand > best:
            best = cand
            best_thr = float(thr)
            best_pack = {
                "Accuracy": acc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
                "ConfusionMatrix": cm.tolist(),
                "Report": rep,
            }
    return best_thr, best_pack


def auto_search_persistence(labels, seg_scores, precision_floor: float, m_grid, k_grid_mode: str):
    """Search over (k,m) and threshold to maximize recall on validation.

    Returns a dict with keys: k, m, thr, pack.
    """
    best = None
    best_sel = None

    for m in m_grid:
        m = int(m)
        if k_grid_mode == "strict":
            k_list = [m, max(1, m - 1)]
        elif k_grid_mode == "medium":
            k_list = [max(1, m - 1), max(1, m - 2)]
        else:
            k_list = [max(1, m - 2), max(1, m - 3)]
        # Clean up duplicates and enforce bounds
        k_list = list(dict.fromkeys([int(min(max(k, 1), m)) for k in k_list]))
        for k in k_list:
            thr, pack = select_persistence_threshold_max_recall(labels, seg_scores, k=k, m=m, precision_floor=precision_floor)
            if pack is None:
                continue
            cand = (
                pack["Recall"],
                pack["Precision"],
                pack["F1"],
                -m,
                -k,
                -thr,
            )
            if best is None or cand > best:
                best = cand
                best_sel = {"k": k, "m": m, "thr": thr, "pack": pack}
    return best_sel


def compute_scores(model, loader, device):
    model.eval()
    assert not model.training, "model must be in eval mode"
    all_video_scores, all_seg_scores, all_labels = [], [], []
    with torch.no_grad():
        for feats, lbl in loader:
            feats = feats.to(device)
            lbl = lbl.to(device)
            video_score, seg_scores, _ = model.infer(feats)
            all_video_scores.append(video_score.detach().cpu().numpy())
            all_seg_scores.append(seg_scores.detach().cpu().numpy())
            all_labels.append(lbl.detach().cpu().numpy())
    video_scores = np.concatenate(all_video_scores, axis=0).reshape(-1).astype(np.float64)
    seg_scores = np.concatenate(all_seg_scores, axis=0).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int32)
    return labels, video_scores, seg_scores


def main():
    # Extend option parser with additional arguments specific to calibration
    parser = argparse.ArgumentParser(description="Threshold calibration and test evaluation for DA-RTFM", add_help=False)
    # Inherit options from the main option parser (feature lists, model hyperparams)
    for action in option_parser._actions:
        if action.option_strings:
            parser._add_action(action)

    parser.add_argument('--model-path', type=str, required=True, help='Path to a trained model checkpoint (.pkl)')
    parser.add_argument('--precision-floor', type=float, default=0.85, help='Minimum precision required during threshold search')
    parser.add_argument('--alert-rule', type=str, default='auto', choices=['video_score', 'segment_persistence', 'auto'],
                        help='Which alerting rule to use for threshold search; "auto" tries both and picks the best on validation')
    parser.add_argument('--m-grid', type=str, default='3,4,5,6,8', help='Comma‑separated list of persistence window lengths to search over')
    parser.add_argument('--k-grid-mode', type=str, default='strict', choices=['strict', 'medium', 'permissive'],
                        help='Strategy for selecting k relative to m when searching persistence rules')
    parser.add_argument('--output-report', type=str, default=None,
                        help='Optional path to write a JSON summary of validation and test metrics')
    parser.add_argument('--threshold-policy', type=str, default='precision_floor',
                        choices=['precision_floor', 'youdens_j'],
                        help='"precision_floor" (default): maximise recall subject to precision >= floor; '
                             '"youdens_j": maximise TPR+TNR-1, ignores --precision-floor entirely')

    args, _ = parser.parse_known_args()

    # Ensure a validation list is provided; calibration requires a held‑out validation set.
    if not getattr(args, 'val_rgb_list', None):
        raise ValueError('Calibration requires a validation list (--val-rgb-list) separate from the training and test lists.')

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine feature dimension
    input_feature_size = args.feature_size + (args.flow_dim if getattr(args, 'use_flow', False) else 0)

    # Instantiate model and load weights
    model = Model(
        n_features=input_feature_size,
        batch_size=args.batch_size,
        num_segments=args.num_segments,
        topk_ratio=getattr(args, 'topk_ratio', 0.25),
    ).to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Load validation set (no label filtering)
    val_args = argparse.Namespace(**vars(args))
    val_args.rgb_list = args.val_rgb_list
    val_args.test_rgb_list = args.val_rgb_list
    val_set = Dataset(val_args, is_normal=True, test_mode=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_labels, val_video_scores, val_seg_scores = compute_scores(model, val_loader, device)

    # Compute threshold‑independent metrics on validation
    val_auc_roc = roc_auc_score(val_labels, val_video_scores) if len(np.unique(val_labels)) > 1 else float('nan')
    val_auc_pr = average_precision_score(val_labels, val_video_scores) if len(np.unique(val_labels)) > 1 else float('nan')

    precision_floor = float(args.precision_floor)
    threshold_policy = getattr(args, 'threshold_policy', 'precision_floor')

    # Threshold search on validation
    chosen = None
    val_alert_summary = {}

    if threshold_policy == 'youdens_j':
        # Youden's J: ignores precision_floor entirely; always succeeds
        yj_thr, yj_pack = select_threshold_youdens_j(val_labels, val_video_scores)
        chosen = {"type": "video_score", "thr": float(yj_thr), "pack": yj_pack,
                  "policy": "youdens_j"}
        print(f"[Youden's J] threshold={yj_thr:.6f}  "
              f"J={yj_pack['Recall'] + (1 - yj_pack.get('Precision', 0)) - 1:.4f}  "
              f"val_F1={yj_pack['F1']:.4f}  val_Acc={yj_pack['Accuracy']:.4f}")
    else:
        # Original precision-floor policy
        if args.alert_rule in ("segment_persistence", "auto"):
            m_grid = [int(x) for x in args.m_grid.split(',') if x.strip()]
            persist_sel = auto_search_persistence(val_labels, val_seg_scores, precision_floor, m_grid, args.k_grid_mode)
            if persist_sel is not None:
                chosen = {"type": "segment_persistence", **persist_sel}
        if args.alert_rule in ("video_score", "auto"):
            vid_thr, vid_pack = select_video_threshold_max_recall(val_labels, val_video_scores, precision_floor)
            if vid_pack is not None:
                vid_choice = {"type": "video_score", "thr": float(vid_thr), "pack": vid_pack}
                if chosen is None:
                    chosen = vid_choice
                else:
                    def rank(d):
                        if d['type'] == 'video_score':
                            p = vid_pack
                            return (p['Recall'], p['Precision'], p['F1'], -d['thr'])
                        else:
                            p = d['pack']
                            return (p['Recall'], p['Precision'], p['F1'], -d['m'], -d['k'], -d['thr'])
                    if rank(vid_choice) > rank(chosen):
                        chosen = vid_choice

    # Prepare summary of validation calibration
    val_alert_summary['AUC_ROC'] = val_auc_roc
    val_alert_summary['AUC_PR'] = val_auc_pr
    if chosen is None:
        raise RuntimeError('No valid threshold found on validation set under precision floor {}'.format(precision_floor))
    val_alert_summary['SelectedRule'] = chosen

    # Apply selected threshold to test set
    # Load test set
    test_args = argparse.Namespace(**vars(args))
    # Ensure that rgb_list points to the test list
    test_args.rgb_list = args.test_rgb_list
    test_args.test_rgb_list = args.test_rgb_list
    test_set = Dataset(test_args, is_normal=True, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_labels, test_video_scores, test_seg_scores = compute_scores(model, test_loader, device)

    # Compute test metrics at the chosen operating point
    if chosen['type'] == 'video_score':
        thr = chosen['thr']
        preds = (test_video_scores >= thr).astype(np.int32)
        acc, f1, prec, rec, cm, rep = metrics(test_labels, preds)
        test_summary = {
            "Rule": "video_score",
            "Threshold": float(thr),
            "Accuracy": acc,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "ConfusionMatrix": cm.tolist(),
            "Report": rep,
        }
    else:
        thr = chosen['thr']
        k = chosen['k']
        m = chosen['m']
        preds = segment_persistence_alert(test_seg_scores, float(thr), k=k, m=m)
        acc, f1, prec, rec, cm, rep = metrics(test_labels, preds)
        test_summary = {
            "Rule": "segment_persistence",
            "Threshold": float(thr),
            "k": k,
            "m": m,
            "Accuracy": acc,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "ConfusionMatrix": cm.tolist(),
            "Report": rep,
        }

    # Print human‑readable summary
    print("=== Validation Summary ===")
    print(f"ROC AUC: {val_auc_roc:.4f} | PR AUC: {val_auc_pr:.4f}")
    print("Chosen rule on validation:", chosen)
    print("=== Test Summary ===")
    for k, v in test_summary.items():
        if k == 'ConfusionMatrix' or k == 'Report':
            continue
        print(f"{k}: {v}")
    print("Confusion Matrix:\n", test_summary['ConfusionMatrix'])
    print("Classification Report:\n", test_summary['Report'])

    # Optionally write JSON report
    if args.output_report:
        report = {
            "validation": val_alert_summary,
            "test": test_summary,
        }
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print("Wrote report to", args.output_report)


if __name__ == '__main__':
    main()