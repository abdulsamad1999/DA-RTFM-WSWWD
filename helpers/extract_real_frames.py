#!/usr/bin/env python3
"""Extract representative frames and score timelines from real-world test videos.

Usage:
    python helpers/extract_real_frames.py \
        --real-list list/splits/real_test_v2.list \
        --runs-root runs_da_rtfm_v3 \
        --output-dir paper_assets/real_frames \
        --n-per-class 4 \
        --use-flow --num-segments 32
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Model  # noqa: E402


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _detect_feat_dim(ckpt_path: Path, use_flow: bool) -> int:
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        for k, v in state.items():
            if "Aggregate" in k and "conv" in k and "weight" in k and v.dim() == 3:
                return int(v.shape[1])
    except Exception:
        pass
    return 1028 if use_flow else 1024


def load_model(runs_root: Path, use_flow: bool, num_segments: int,
               device: torch.device) -> Model:
    """Load seed-42 (first alphabetically sorted) DA-RTFM checkpoint."""
    runs = sorted(runs_root.glob("*_rtfm_wd*"))
    if not runs:
        runs = sorted(r for r in runs_root.iterdir() if r.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories in {runs_root}")
    run = runs[0]
    ckpts = sorted((run / "ckpt").glob("*_best.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"No *_best.pkl in {run}/ckpt/")
    ckpt_path = ckpts[-1]

    feat_dim = _detect_feat_dim(ckpt_path, use_flow)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = Model(n_features=feat_dim, batch_size=1,
                  topk_ratio=0.25, num_segments=num_segments)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  Loaded: {ckpt_path.name}  feat_dim={feat_dim}")
    return model


def load_calibrated_threshold(runs_root: Path) -> float:
    """Return calibrated threshold from first run's calibration_report.json."""
    for run in sorted(runs_root.glob("*_rtfm_wd*")):
        cal = run / "metrics" / "calibration_report.json"
        if cal.exists():
            data = json.loads(cal.read_text(encoding="utf-8"))
            thr = float(data["validation"]["SelectedRule"]["thr"])
            print(f"  Calibrated threshold: {thr:.4f}  (from {cal.name})")
            return thr
    print("  No calibration_report.json found; using threshold=0.5")
    return 0.5


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def feat_path_to_video(feat_path: str, dataset_root: Path) -> Path:
    """Map features_v2/real-only_I3D/Class/file.npy -> dataset/real-only_I3D/Class/file.mp4"""
    p = Path(feat_path)
    for i, part in enumerate(p.parts):
        if part in ("real-only_I3D", "testing_I3D"):
            rel = Path(*p.parts[i:]).with_suffix(".mp4")
            return dataset_root / rel
    return dataset_root / p.with_suffix(".mp4").name


def extract_frame_at_pct(video_path: Path, pct: float = 0.45) -> Optional[np.ndarray]:
    """Seek to `pct` of total frames and read one frame. Returns BGR or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, min(int(total * pct), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_video(feat_path: str, use_flow: bool,
                model: Model, device: torch.device) -> np.ndarray:
    """Load feature (+flow), run model.infer(). Returns seg_scores (T,)."""
    feat = np.load(feat_path, allow_pickle=True).astype(np.float32)
    if use_flow:
        flow_p = feat_path.replace(".npy", "_flow.npy")
        if Path(flow_p).exists():
            flow = np.load(flow_p, allow_pickle=True).astype(np.float32)
            feat = np.concatenate([feat, flow], axis=1)
    t = torch.from_numpy(feat[np.newaxis]).to(device)
    with torch.no_grad():
        _, seg_scores, _ = model.infer(t)
    return seg_scores.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting (bar chart style matching visualize_errors.py)
# ---------------------------------------------------------------------------

def plot_score_bar(seg_scores: np.ndarray, title: str, out_path: Path,
                   threshold: float) -> None:
    """Bar chart: red bars >= threshold, green bars < threshold, blue dashed thr line."""
    T = len(seg_scores)
    colors = ["red" if s >= threshold else "green" for s in seg_scores]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(range(T), seg_scores, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(y=threshold, color="blue", linestyle="--", linewidth=1.5,
               label=f"thr={threshold:.3f}")

    legend_patches = [
        mpatches.Patch(facecolor="red",   label=f"score >= {threshold:.3f}"),
        mpatches.Patch(facecolor="green", label=f"score < {threshold:.3f}"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Segment", fontsize=9)
    ax.set_ylabel("Anomaly score", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-list",    default="list/splits/real_test_v2.list")
    parser.add_argument("--runs-root",    default="runs_da_rtfm_v3")
    parser.add_argument("--output-dir",   default="paper_assets/real_frames")
    parser.add_argument("--n-per-class",  type=int, default=4)
    parser.add_argument("--use-flow",     action="store_true")
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--dataset-root", default=None,
                        help="Root of dataset/ folder (auto-detected if omitted)")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_root  = REPO_ROOT / args.runs_root
    out_dir    = REPO_ROOT / args.output_dir

    dataset_root = (
        Path(args.dataset_root) if args.dataset_root
        else REPO_ROOT.parent / "dataset"
    )
    if not dataset_root.is_dir():
        dataset_root = REPO_ROOT / "dataset"
    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {out_dir}")

    for sub in ("Normal", "Anomaly", "scores"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Load list
    lines = [ln.strip() for ln in
             open(REPO_ROOT / args.real_list, encoding="utf-8") if ln.strip()]
    normal_lines  = [ln for ln in lines if ln.split()[-1] == "0"]
    anomaly_lines = [ln for ln in lines if ln.split()[-1] == "1"]
    print(f"Real test    : {len(normal_lines)}N + {len(anomaly_lines)}A")

    # Load model + threshold
    model     = load_model(runs_root, args.use_flow, args.num_segments, device)
    threshold = load_calibrated_threshold(runs_root)

    saved_files: list = []

    for class_name, class_lines, label in [
        ("Normal",  normal_lines,  0),
        ("Anomaly", anomaly_lines, 1),
    ]:
        n = len(class_lines)
        # Evenly-spread: indices at 1/5, 2/5, 3/5, 4/5 of the list
        indices = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5]
        indices = list(dict.fromkeys(max(0, min(i, n - 1)) for i in indices))
        indices = indices[: args.n_per_class]

        prefix = "Real_Normal" if label == 0 else "Real_Wrongway"

        for out_idx, src_idx in enumerate(indices, start=1):
            feat_path  = class_lines[src_idx].split()[0]
            video_path = feat_path_to_video(feat_path, dataset_root)

            print(f"\n[{class_name} {out_idx}/{args.n_per_class}]  {feat_path}")
            print(f"  Video : {video_path}")

            # ── Frame extraction ──
            frame_name = f"{prefix}_frame_{out_idx:03d}.jpg"
            frame_out  = out_dir / class_name / frame_name
            if video_path.exists():
                frame = extract_frame_at_pct(video_path, pct=0.45)
                if frame is not None:
                    cv2.imwrite(str(frame_out), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    h, w = frame.shape[:2]
                    print(f"  Frame : {frame_out}  ({w}x{h})")
                    saved_files.append(str(frame_out))
                else:
                    print(f"  [WARN] Could not read frame from {video_path}")
            else:
                print(f"  [WARN] Video not found: {video_path}")

            # ── Score timeline ──
            score_name = f"{prefix}_scores_{out_idx:03d}.png"
            score_out  = out_dir / "scores" / score_name
            try:
                seg_scores  = score_video(str(REPO_ROOT / feat_path),
                                          args.use_flow, model, device)
                video_score = float(seg_scores.max())
                pred_label  = "ALERT" if video_score >= threshold else "OK"
                title = (f"{prefix} #{out_idx}  "
                         f"max={video_score:.3f}  thr={threshold:.3f}  [{pred_label}]")
                plot_score_bar(seg_scores, title, score_out, threshold)
                print(f"  Score : max={video_score:.3f}  pred={pred_label}")
                print(f"  Plot  : {score_out}")
                saved_files.append(str(score_out))
            except Exception as exc:
                print(f"  [WARN] Scoring failed: {exc}")

    print("\n=== Saved files ===")
    for f in saved_files:
        print(f"  {f}")
    print(f"\nTotal     : {len(saved_files)} files")
    print(f"Output dir: {out_dir}")
    print(f"Threshold : {threshold:.4f}")


if __name__ == "__main__":
    main()
