#!/usr/bin/env python3
"""Feature extraction for DA-RTFM — CUDA-accelerated, batched segment inference.

Scans --video-root for all .mp4 files under Normal/ and Anomaly/ subdirectories,
extracts I3D RGB features (and optional optical-flow statistics), and writes
.npy files to --output-root preserving the same subdirectory layout.

Resample modes
--------------
adaptive (default):
    Divide the video's actual frames into T=32 equal-duration bins via numpy
    linspace.  Each bin yields exactly 16 frames (repeating frames if the bin
    has fewer than 16).  Bins with fewer than 4 original frames trigger a
    warning; their frames are tiled to at least 4 before linspace sampling.
    No zero-padding: every feature vector carries real content.

pad (backward-compatible):
    The entire video is forwarded through I3D as one long clip, yielding T'
    temporal feature vectors which are then adaptive-pooled to num_segments.
    Behaves identically to the original extract_features.py.

Batched inference
-----------------
In adaptive mode all T segments are packed into a single tensor
(T, 3, 16, H, W) and sent to GPU in one forward pass — far more efficient
than T separate calls.

Usage
-----
    python helpers/extract_features.py \\
        --video-root dataset/training_I3D \\
        --output-root features_v2/training_I3D \\
        --num-segments 32 \\
        --resample-mode adaptive \\
        --use-flow \\
        --save-theta results/theta_ref_v2.json

    python helpers/extract_features.py \\
        --video-root dataset/testing_I3D \\
        --output-root features_v2/testing_I3D \\
        --num-segments 32 \\
        --resample-mode adaptive \\
        --use-flow \\
        --theta-ref results/theta_ref_v2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pytorch_i3d import InceptionI3d  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract I3D RGB+flow features from videos")
    p.add_argument("--video-root", required=True,
                   help="Root directory with Normal/ and Anomaly/ sub-folders containing .mp4 files")
    p.add_argument("--output-root", required=True,
                   help="Destination root for .npy feature files (same sub-folder layout)")
    p.add_argument("--i3d-checkpoint", default="rgb_imagenet.pt",
                   help="Path to pretrained I3D RGB weights (default: rgb_imagenet.pt)")
    p.add_argument("--num-segments", type=int, default=32,
                   help="Number of temporal segments (default 32)")
    p.add_argument("--frames-per-segment", type=int, default=16,
                   help="Frames fed to I3D per segment clip (must be >=16; default 16)")
    p.add_argument("--resample-mode", default="adaptive", choices=["adaptive", "pad"],
                   help="adaptive=bin+repeat (default); pad=whole-video then pool (legacy)")
    p.add_argument("--use-flow", action="store_true",
                   help="Also compute per-segment optical-flow statistics [u, v, mag, D_t]")
    p.add_argument("--flow-suffix", default="_flow",
                   help="Suffix for flow sidecar files (default: _flow)")
    p.add_argument("--save-theta", default=None,
                   help="Save estimated theta_ref to this JSON path (training only)")
    p.add_argument("--theta-ref", default=None,
                   help="Load pre-computed theta_ref from this JSON path (testing)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip videos whose .npy output already exists")
    p.add_argument("--seed", type=int, default=123, help="Random seed (unused, kept for compat)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR frame -> normalized (3, 224, 224) float32 tensor."""
    frame = cv2.resize(frame_bgr, (340, 256), interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]
    top, left = (h - 224) // 2, (w - 224) // 2
    crop = frame[top:top + 224, left:left + 224]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    return torch.from_numpy(np.transpose(crop, (2, 0, 1)))  # (3, 224, 224)


def load_all_frames(video_path: str) -> List[torch.Tensor]:
    """Load and preprocess every frame from the video. Returns list of (3,224,224) tensors."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(preprocess_frame(frame))
        ok, frame = cap.read()
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Adaptive bin construction
# ---------------------------------------------------------------------------

def bin_to_clip(frames: List[torch.Tensor], start: int, end: int,
                frames_per_seg: int, video_path: str, bin_idx: int) -> torch.Tensor:
    """Extract exactly `frames_per_seg` frames from frames[start:end], repeating if needed."""
    bin_frames = frames[start:end]
    if len(bin_frames) == 0:
        # Edge case: empty bin — grab nearest frame
        idx = max(0, start - 1)
        bin_frames = [frames[idx]]

    if len(bin_frames) < 4:
        print(f"  [WARN] bin {bin_idx} has {len(bin_frames)} frame(s) in {Path(video_path).name}"
              f" (start={start}, end={end}); tiling to >=4.", flush=True)
        while len(bin_frames) < 4:
            bin_frames = bin_frames + bin_frames

    # linspace-sample exactly frames_per_seg indices (repeats if len < frames_per_seg)
    idxs = np.linspace(0, len(bin_frames) - 1, frames_per_seg, dtype=int)
    clip = torch.stack([bin_frames[i] for i in idxs], dim=1)  # (3, frames_per_seg, 224, 224)
    return clip


# ---------------------------------------------------------------------------
# RGB feature extraction
# ---------------------------------------------------------------------------

def extract_rgb_adaptive(
    video_path: str,
    model: InceptionI3d,
    num_segments: int,
    frames_per_seg: int,
    device: torch.device,
) -> np.ndarray:
    """Adaptive mode: one GPU forward pass over all segments. Returns (T, 1024)."""
    frames = load_all_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    n = len(frames)
    bins = np.linspace(0, n, num_segments + 1, dtype=int)
    clips = [
        bin_to_clip(frames, int(bins[i]), int(bins[i + 1]), frames_per_seg, video_path, i)
        for i in range(num_segments)
    ]
    # batch: (T, 3, frames_per_seg, 224, 224)
    batch = torch.stack(clips, dim=0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        feats = model.extract_features(batch)   # (T, 1024, 1, 1, 1)
    feats = feats.squeeze(-1).squeeze(-1).squeeze(-1).float().cpu().numpy()  # (T, 1024)

    # Sanity check: no near-zero feature vectors
    norms = np.linalg.norm(feats, axis=1)
    bad = np.where(norms < 0.001)[0]
    for b in bad:
        print(f"  [WARN] near-zero feature at segment {b} in {Path(video_path).name} "
              f"(L2={norms[b]:.6f})", flush=True)

    return feats.astype(np.float32)


def extract_rgb_pad(
    video_path: str,
    model: InceptionI3d,
    num_segments: int,
    device: torch.device,
) -> np.ndarray:
    """Pad mode (legacy): forward entire video through I3D, then adaptive-pool. Returns (T, 1024)."""
    frames = load_all_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    clip = torch.stack(frames, dim=1).unsqueeze(0).to(device)  # (1, 3, N, 224, 224)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        raw = model.extract_features(clip)  # (1, 1024, T', 1, 1)
    raw = raw.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0).float().cpu().numpy()  # (T', 1024)

    from feature_utils import adaptive_pool
    return adaptive_pool(raw, num_segments).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical-flow features
# ---------------------------------------------------------------------------

def compute_flow_features(
    video_path: str,
    num_segments: int,
    theta_ref: Optional[float],
    is_flip: bool,
    flow_size: int = 128,
) -> np.ndarray:
    """Compute per-segment [u, v, mag, D_t] flow features. Returns (T, 4).

    Frames are resized to flow_size×flow_size before Farneback to keep
    computation tractable on CPU regardless of source resolution.
    """
    cap = cv2.VideoCapture(video_path)
    gray_frames = []
    ok, frame = cap.read()
    while ok:
        small = cv2.resize(frame, (flow_size, flow_size), interpolation=cv2.INTER_LINEAR)
        gray_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        ok, frame = cap.read()
    cap.release()

    n = len(gray_frames)
    if n < 2:
        raise RuntimeError(f"Not enough frames for flow in {video_path}")

    bins = np.linspace(0, n - 1, num_segments + 1, dtype=int)
    theta_use = theta_ref
    if is_flip and theta_use is not None:
        theta_use = float(np.pi) - theta_use

    flow_feats = []
    for s in range(num_segments):
        start, end = int(bins[s]), int(bins[s + 1])
        end = min(end, n - 1)
        u_vals, v_vals, mag_vals, theta_vals = [], [], [], []
        for t in range(start, end):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t], gray_frames[t + 1], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            u, v = flow[..., 0], flow[..., 1]
            u_vals.append(float(np.mean(u)))
            v_vals.append(float(np.mean(v)))
            mag_vals.append(float(np.mean(np.sqrt(u ** 2 + v ** 2))))
            theta_vals.append(float(np.mean(np.arctan2(v, u))))

        if not u_vals:
            row = [0.0, 0.0, 0.0, 0.0]
        else:
            u_m = float(np.mean(u_vals))
            v_m = float(np.mean(v_vals))
            mag  = float(np.mean(mag_vals))
            th   = float(np.mean(theta_vals))
            D_t  = float(np.cos(th - theta_use)) if theta_use is not None else 1.0
            row  = [u_m, v_m, mag, D_t]
        flow_feats.append(row)

    return np.array(flow_feats, dtype=np.float32)


def estimate_theta_ref(
    normal_videos: List[Path],
    num_segments: int,
    max_videos: int = 50,
    flow_size: int = 128,
) -> float:
    """Estimate dominant traffic direction from normal non-flip videos (circular mean).

    Single-pass: opens each video once, resizes frames to flow_size for speed,
    and accumulates raw arctan2 theta values across all segment bins.
    """
    theta_values: List[float] = []
    processed = 0
    for vp in normal_videos:
        if "_flip_" in vp.stem:
            continue
        cap = cv2.VideoCapture(str(vp))
        gray: List[np.ndarray] = []
        ok, frame = cap.read()
        while ok:
            small = cv2.resize(frame, (flow_size, flow_size), interpolation=cv2.INTER_LINEAR)
            gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            ok, frame = cap.read()
        cap.release()
        n = len(gray)
        if n < 2:
            continue
        bins = np.linspace(0, n - 1, num_segments + 1, dtype=int)
        for s in range(num_segments):
            start, end = int(bins[s]), min(int(bins[s + 1]), n - 1)
            for t in range(start, end):
                flow = cv2.calcOpticalFlowFarneback(
                    gray[t], gray[t + 1], None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0,
                )
                theta_values.append(float(np.mean(np.arctan2(flow[..., 1], flow[..., 0]))))
        processed += 1
        if processed >= max_videos:
            break

    if not theta_values:
        raise RuntimeError("No theta values found in normal training videos")

    sin_m = float(np.mean(np.sin(theta_values)))
    cos_m = float(np.mean(np.cos(theta_values)))
    theta_ref = float(np.arctan2(sin_m, cos_m))
    print(f"[INFO] theta_ref={theta_ref:.4f} rad estimated from "
          f"{processed} normal videos ({len(theta_values)} flow measurements)", flush=True)
    return theta_ref


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_videos(video_root: str) -> List[Tuple[Path, int]]:
    """Return [(video_path, label), ...] for all .mp4 files under Normal/ and Anomaly/."""
    root = Path(video_root)
    items: List[Tuple[Path, int]] = []
    for cls_name, label in [("Normal", 0), ("Anomaly", 1)]:
        cls_dir = root / cls_name
        if cls_dir.is_dir():
            for vp in sorted(cls_dir.glob("*.mp4")):
                items.append((vp, label))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- CUDA setup --------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available — running on CPU (will be slow)", flush=True)

    # ---- Load I3D model ----------------------------------------------------
    ckpt = Path(args.i3d_checkpoint)
    if not ckpt.exists():
        # Try repo root as fallback
        ckpt = REPO_ROOT / args.i3d_checkpoint
    if not ckpt.exists():
        sys.exit(f"ERROR: I3D checkpoint not found: {args.i3d_checkpoint}")

    print(f"[INFO] Loading I3D weights from {ckpt}", flush=True)
    i3d = InceptionI3d(400, in_channels=3)
    state_dict = torch.load(str(ckpt), map_location=device)
    i3d.load_state_dict(state_dict)
    i3d.replace_logits(1024)
    i3d.to(device).eval()
    print(f"[INFO] I3D loaded on {device}", flush=True)
    if device.type == "cuda":
        print(f"[INFO] GPU memory after model load: "
              f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)

    # ---- Scan videos -------------------------------------------------------
    items = scan_videos(args.video_root)
    if not items:
        sys.exit(f"ERROR: No .mp4 files found under {args.video_root}")
    print(f"[INFO] Found {len(items)} videos in {args.video_root}", flush=True)

    output_root = Path(args.output_root)

    # ---- Determine theta_ref -----------------------------------------------
    theta_ref: Optional[float] = None
    if args.use_flow:
        if args.theta_ref:
            with open(args.theta_ref, "r", encoding="utf-8") as f:
                data = json.load(f)
            theta_ref = float(data["theta_ref"])
            print(f"[INFO] Loaded theta_ref={theta_ref:.6f} from {args.theta_ref}", flush=True)
        elif args.save_theta:
            print("[INFO] Estimating theta_ref from normal non-flip videos ...", flush=True)
            normal_videos = [vp for vp, lbl in items if lbl == 0 and "_flip_" not in vp.stem]
            theta_ref = estimate_theta_ref(normal_videos, args.num_segments)
            Path(args.save_theta).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_theta, "w", encoding="utf-8") as f:
                json.dump({"theta_ref": theta_ref}, f, indent=2)
            print(f"[INFO] Saved theta_ref to {args.save_theta}", flush=True)
        else:
            print("[WARN] --use-flow requested but no --theta-ref / --save-theta provided. "
                  "D_t will be set to 1.0 for all segments.", flush=True)

    # ---- Extract features --------------------------------------------------
    saved = skipped = errors = 0
    total_ms = 0.0

    for idx, (video_path, label) in enumerate(items):
        # Determine output paths
        cls_name = "Anomaly" if label == 1 else "Normal"
        out_rgb  = output_root / cls_name / f"{video_path.stem}.npy"
        out_flow = output_root / cls_name / f"{video_path.stem}{args.flow_suffix}.npy"

        if args.skip_existing and out_rgb.exists():
            if not args.use_flow or out_flow.exists():
                skipped += 1
                continue

        t0 = time.perf_counter()

        try:
            # RGB features
            if args.resample_mode == "adaptive":
                rgb_feats = extract_rgb_adaptive(
                    str(video_path), i3d, args.num_segments,
                    args.frames_per_segment, device,
                )
            else:
                rgb_feats = extract_rgb_pad(
                    str(video_path), i3d, args.num_segments, device,
                )

            # Flow features
            flow_feats = None
            if args.use_flow:
                is_flip = "_flip_" in video_path.stem
                flow_feats = compute_flow_features(
                    str(video_path), args.num_segments,
                    theta_ref, is_flip,
                )

        except Exception as exc:
            print(f"[ERROR] {video_path.name}: {exc}", flush=True)
            errors += 1
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms

        # Write outputs
        out_rgb.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_rgb), rgb_feats)
        if flow_feats is not None:
            np.save(str(out_flow), flow_feats)

        saved += 1
        # Per-video log
        cap_tmp = cv2.VideoCapture(str(video_path))
        n_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid  = cap_tmp.get(cv2.CAP_PROP_FPS) or 0
        dur_s    = n_frames / fps_vid if fps_vid > 0 else 0
        cap_tmp.release()
        print(
            f"[{idx+1:4d}/{len(items)}] {video_path.name}  "
            f"dur={dur_s:.1f}s  frames={n_frames}  "
            f"mode={args.resample_mode}  t={elapsed_ms:.0f}ms",
            flush=True,
        )

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f"Saved   : {saved}", flush=True)
    print(f"Skipped : {skipped}", flush=True)
    print(f"Errors  : {errors}", flush=True)
    avg_ms = total_ms / max(saved, 1)
    print(f"Total   : {total_ms/1000:.1f}s  (avg {avg_ms:.0f}ms/video)", flush=True)
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory: {peak_mb:.1f} MB", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
