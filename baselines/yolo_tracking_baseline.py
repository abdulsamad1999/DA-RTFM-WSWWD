#!/usr/bin/env python3
"""
YOLO + centroid tracking baseline for wrong-way driving detection.

Operates on RAW VIDEO FILES (not pre-extracted features).
Requires: ultralytics (pip install ultralytics)

If ultralytics is not installed or raw videos are not found,
the script exits with clear instructions and code 2.

Algorithm:
1. For each video: detect vehicles per frame with YOLOv8n
2. Track detections using a centroid tracker (IoU-based assignment)
3. For each track with >=5 detections: compute mean displacement vector
4. Compare to dominant traffic direction theta_ref
5. Video anomaly score = fraction of frames containing a track
   whose cosine similarity to theta_ref is < -0.2
6. Threshold on val set, evaluate on test set

theta_ref is estimated from normal training videos (saved to --theta-ref-json).

Usage:
    python baselines/yolo_tracking_baseline.py \
        --test-list list/splits/test_base.list \
        --val-list list/splits/val_base.list \
        --train-list list/splits/train_grouped.list \
        --video-root dataset \
        --precision-floor 0.9 \
        --output-json results/yolo_tracking_baseline.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Centroid Tracker (IoU-based, no external dependencies)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """IoU-based centroid tracker. No external dependencies beyond numpy."""

    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.objects: Dict[int, dict] = {}  # id -> {centroid, box, disappeared, history}

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union = areaA + areaB - inter_area
        return inter_area / (union + 1e-6)

    def _centroid(self, box: List[float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, boxes: List[List[float]]) -> Dict[int, Tuple[float, float]]:
        """Update tracker with new detections.

        Args:
            boxes: List of [x1,y1,x2,y2] detection boxes.

        Returns:
            Dict mapping track_id -> (cx, cy) centroid.
        """
        if not boxes:
            # Mark all as disappeared
            for tid in list(self.objects.keys()):
                self.objects[tid]["disappeared"] += 1
                if self.objects[tid]["disappeared"] > self.max_disappeared:
                    del self.objects[tid]
            return {tid: obj["centroid"] for tid, obj in self.objects.items()}

        # If no existing tracks, register all
        if not self.objects:
            for box in boxes:
                self._register(box)
            return {tid: obj["centroid"] for tid, obj in self.objects.items()}

        # Match detections to existing tracks by IoU
        existing_ids = list(self.objects.keys())
        existing_boxes = [self.objects[tid]["box"] for tid in existing_ids]

        iou_matrix = np.zeros((len(existing_ids), len(boxes)), dtype=np.float32)
        for i, ebox in enumerate(existing_boxes):
            for j, dbox in enumerate(boxes):
                iou_matrix[i, j] = self._iou(ebox, dbox)

        # Greedy matching: highest IoU first
        matched_exist = set()
        matched_det = set()
        for _ in range(min(len(existing_ids), len(boxes))):
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i, j = int(idx[0]), int(idx[1])
            if iou_matrix[i, j] < self.iou_threshold:
                break
            tid = existing_ids[i]
            self._update_track(tid, boxes[j])
            matched_exist.add(i)
            matched_det.add(j)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # Handle unmatched existing tracks
        for i, tid in enumerate(existing_ids):
            if i not in matched_exist:
                self.objects[tid]["disappeared"] += 1
                if self.objects[tid]["disappeared"] > self.max_disappeared:
                    del self.objects[tid]

        # Register new detections
        for j in range(len(boxes)):
            if j not in matched_det:
                self._register(boxes[j])

        return {tid: obj["centroid"] for tid, obj in self.objects.items()}

    def _register(self, box: List[float]) -> None:
        cx, cy = self._centroid(box)
        self.objects[self.next_id] = {
            "centroid": (cx, cy),
            "box": box,
            "disappeared": 0,
            "history": [(cx, cy)],
        }
        self.next_id += 1

    def _update_track(self, tid: int, box: List[float]) -> None:
        cx, cy = self._centroid(box)
        self.objects[tid]["centroid"] = (cx, cy)
        self.objects[tid]["box"] = box
        self.objects[tid]["disappeared"] = 0
        self.objects[tid]["history"].append((cx, cy))

    def get_track_vectors(self, min_len: int = 5) -> Dict[int, Tuple[float, float]]:
        """Compute mean displacement vector for tracks with >= min_len points.

        Returns:
            Dict mapping track_id -> (dx, dy) mean displacement vector.
        """
        vectors = {}
        for tid, obj in self.objects.items():
            hist = obj["history"]
            if len(hist) < min_len:
                continue
            pts = np.array(hist, dtype=np.float64)
            disps = pts[1:] - pts[:-1]
            vectors[tid] = (float(disps[:, 0].mean()), float(disps[:, 1].mean()))
        return vectors


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _check_ultralytics():
    """Check if ultralytics is installed; exit with code 2 if not."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: ultralytics not installed. Run:", file=sys.stderr)
        print("    pip install ultralytics", file=sys.stderr)
        sys.exit(2)


def _load_yolo_model(device: str = "cpu"):
    """Load YOLOv8n model."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.to(device)
    return model


def _feature_path_to_video_path(feature_path: str, video_root: str) -> Optional[Path]:
    """Derive video path from feature path.

    Feature path: features/testing_I3D/Anomaly/Wrongway129_x264.npy
    Video path:   <video_root>/testing_I3D/Anomaly/Wrongway129_x264.mp4

    Also tries the direct mapping: <video_root>/Anomaly/Wrongway129_x264.mp4
    and: <video_root>/Wrongway129_x264.mp4
    """
    fp = Path(feature_path)
    stem = fp.stem  # e.g. Wrongway129_x264

    # Strategy 1: replace 'features/' prefix with video_root, change ext to .mp4
    parts = fp.parts
    try:
        feat_idx = next(i for i, p in enumerate(parts) if p == "features")
        rel_parts = parts[feat_idx + 1:]
        candidate = Path(video_root).joinpath(*rel_parts).with_suffix(".mp4")
        if candidate.exists():
            return candidate
    except StopIteration:
        pass

    # Strategy 2: search recursively
    root = Path(video_root)
    for candidate in root.rglob(f"{stem}.mp4"):
        return candidate

    return None


def estimate_theta_ref(
    normal_video_paths: List[Path],
    yolo_model,
    device: str = "cpu",
    max_videos: int = 20,
) -> float:
    """Estimate dominant traffic direction theta_ref from normal training videos.

    Returns the median angle (in radians) of all track displacement vectors.
    """
    import cv2

    all_angles = []
    for video_path in normal_video_paths[:max_videos]:
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        tracker = CentroidTracker(max_disappeared=10, iou_threshold=0.3)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 3 == 0:  # Sample every 3rd frame for speed
                results = yolo_model(frame, classes=[2, 5, 7], verbose=False)  # car, bus, truck
                boxes = []
                for r in results:
                    for box in r.boxes.xyxy.cpu().numpy():
                        boxes.append(box.tolist())
                tracker.update(boxes)
            frame_idx += 1
        cap.release()

        vectors = tracker.get_track_vectors(min_len=5)
        for dx, dy in vectors.values():
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                all_angles.append(math.atan2(dy, dx))

    if not all_angles:
        print("WARNING: No track vectors found in normal videos; using theta_ref=0.0")
        return 0.0

    # Use median direction
    theta_ref = float(np.median(all_angles))
    print(f"Estimated theta_ref={theta_ref:.4f} rad from {len(all_angles)} track vectors")
    return theta_ref


def score_video(
    video_path: Path,
    yolo_model,
    tracker_cls,
    theta_ref: float,
    wrong_way_cos_threshold: float = -0.2,
    device: str = "cpu",
) -> float:
    """Compute video anomaly score: fraction of frames containing a wrong-way track.

    Args:
        video_path: Path to the video file.
        yolo_model: Loaded YOLO model.
        tracker_cls: CentroidTracker class.
        theta_ref: Reference direction angle (radians).
        wrong_way_cos_threshold: Frames where cosine similarity < this are flagged.
        device: Device string (unused here, model already on device).

    Returns:
        Float score in [0, 1].
    """
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    tracker = tracker_cls(max_disappeared=10, iou_threshold=0.3)
    total_frames = 0
    flagged_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        if frame_idx % 2 == 0:  # Sample every other frame
            results = yolo_model(frame, classes=[2, 5, 7], verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    boxes.append(box.tolist())
            tracker.update(boxes)

            # Check for wrong-way tracks in this frame
            ref_vec = np.array([math.cos(theta_ref), math.sin(theta_ref)], dtype=np.float64)
            frame_flagged = False
            for tid, obj in tracker.objects.items():
                hist = obj["history"]
                if len(hist) < 3:
                    continue
                pts = np.array(hist[-5:], dtype=np.float64)
                disps = pts[1:] - pts[:-1]
                mean_disp = disps.mean(axis=0)
                mag = np.linalg.norm(mean_disp)
                if mag < 1.0:
                    continue
                cos_sim = float(np.dot(mean_disp / mag, ref_vec))
                if cos_sim < wrong_way_cos_threshold:
                    frame_flagged = True
                    break
            if frame_flagged:
                flagged_frames += 1

        frame_idx += 1

    cap.release()
    if total_frames == 0:
        return 0.0
    return flagged_frames / total_frames


def load_list(path: str) -> List[Tuple[str, int]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items.append((parts[0], int(float(parts[1]))))
    return items


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


def run_baseline(
    test_list: str,
    val_list: str,
    train_list: str,
    video_root: str,
    theta_ref_json: str,
    precision_floor: float,
    output_json: str,
    device: str = "cpu",
) -> None:
    """Full YOLO+tracking baseline evaluation pipeline."""
    _check_ultralytics()
    print("Loading YOLOv8n model...")
    yolo_model = _load_yolo_model(device=device)

    # Estimate or load theta_ref
    theta_ref_path = Path(theta_ref_json)
    if theta_ref_path.exists():
        with theta_ref_path.open("r", encoding="utf-8") as f:
            theta_ref_data = json.load(f)
        theta_ref = float(theta_ref_data["theta_ref"])
        print(f"Loaded theta_ref={theta_ref:.4f} from {theta_ref_path}")
    else:
        print("Estimating theta_ref from normal training videos...")
        train_items = load_list(train_list)
        normal_items = [(p, l) for p, l in train_items if l == 0]
        normal_video_paths = []
        for feat_path, _ in normal_items:
            vp = _feature_path_to_video_path(feat_path, video_root)
            if vp:
                normal_video_paths.append(vp)
        if not normal_video_paths:
            print("WARNING: No normal training videos found. Using theta_ref=0.0", file=sys.stderr)
            theta_ref = 0.0
        else:
            theta_ref = estimate_theta_ref(normal_video_paths, yolo_model, device=device)
        theta_ref_path.parent.mkdir(parents=True, exist_ok=True)
        with theta_ref_path.open("w", encoding="utf-8") as f:
            json.dump({"theta_ref": theta_ref, "n_normal_videos": len(normal_video_paths)}, f)
        print(f"Saved theta_ref to {theta_ref_path}")

    def score_items(items: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
        scores_list = []
        labels_list = []
        skipped = 0
        for feat_path, label in items:
            vp = _feature_path_to_video_path(feat_path, video_root)
            if vp is None or not vp.exists():
                print(f"  SKIP (video not found): {feat_path}")
                skipped += 1
                continue
            try:
                s = score_video(vp, yolo_model, CentroidTracker, theta_ref, device=device)
                scores_list.append(s)
                labels_list.append(label)
            except Exception as e:
                print(f"  SKIP ({type(e).__name__}): {vp}: {e}")
                skipped += 1
        return np.array(scores_list), np.array(labels_list), skipped

    # Score val set for threshold
    print("Scoring validation set...")
    val_items = load_list(val_list)
    val_scores, val_labels, val_skipped = score_items(val_items)
    print(f"Val: {len(val_scores)} scored, {val_skipped} skipped")

    if len(val_scores) == 0:
        print("ERROR: No val videos scored. Check --video-root.", file=sys.stderr)
        sys.exit(1)

    threshold = select_threshold(val_labels, val_scores, precision_floor)
    print(f"Selected threshold={threshold:.6f}")

    # Score test set
    print("Scoring test set...")
    test_items = load_list(test_list)
    test_scores, test_labels, test_skipped = score_items(test_items)
    print(f"Test: {len(test_scores)} scored, {test_skipped} skipped")

    if len(test_scores) == 0:
        print("ERROR: No test videos scored.", file=sys.stderr)
        sys.exit(1)

    if test_skipped / max(len(test_items), 1) > 0.5:
        print(
            "ERROR: More than 50% of test videos not found. Check --video-root.",
            file=sys.stderr,
        )
        sys.exit(1)

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
    acc = float(accuracy_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, zero_division=0))
    prec = float(precision_score(test_labels, preds, zero_division=0))
    rec = float(recall_score(test_labels, preds, zero_division=0))
    cm = confusion_matrix(test_labels, preds).tolist()
    rep = classification_report(test_labels, preds, digits=4)

    print(f"\nYOLO+Tracking Baseline Results (threshold={threshold:.6f}):")
    print(f"  Test AUC-ROC : {test_auc_roc:.4f}")
    print(f"  Test AUC-PR  : {test_auc_pr:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1           : {f1:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")

    result = {
        "method": "yolo_tracking",
        "theta_ref": theta_ref,
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
        "val_skipped": val_skipped,
        "test_skipped": test_skipped,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO + centroid tracking baseline for wrong-way driving detection"
    )
    parser.add_argument("--test-list", required=True, help="Test manifest (base scenes only)")
    parser.add_argument("--val-list", required=True, help="Validation manifest for threshold selection")
    parser.add_argument("--train-list", required=True, help="Training manifest (for theta_ref estimation)")
    parser.add_argument("--video-root", default="dataset", help="Root directory of raw video files")
    parser.add_argument(
        "--theta-ref-json",
        default="results/yolo_theta_ref.json",
        help="Path to save/load theta_ref estimate",
    )
    parser.add_argument("--precision-floor", type=float, default=0.9)
    parser.add_argument(
        "--output-json",
        default="results/yolo_tracking_baseline.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for YOLO inference (default: cpu to fit alongside feature extraction)",
    )
    args = parser.parse_args()

    run_baseline(
        test_list=args.test_list,
        val_list=args.val_list,
        train_list=args.train_list,
        video_root=args.video_root,
        theta_ref_json=args.theta_ref_json,
        precision_floor=args.precision_floor,
        output_json=args.output_json,
        device=args.device,
    )


if __name__ == "__main__":
    main()
