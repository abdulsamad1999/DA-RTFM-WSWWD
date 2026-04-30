#!/usr/bin/env python3
"""Diagnose whether the D_t (direction cosine) column in flow features is informative.

Loads a sample of feature files that have flow sidecars and reports
the distribution of D_t (last column of the 4-D flow descriptor).
A healthy distribution has values spread across [-1, +1].
A degenerate distribution (all near +1.0 or all near -1.0) indicates
the direction signal provides no discriminative information.

Output saved to --output-json.

Usage:
    python helpers/diagnose_flow.py \
        --test-list list/splits/test_base.list \
        --flow-suffix _flow --sample 40 \
        --output-json results/flow_diagnostic.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from feature_utils import flow_sidecar_path  # noqa: E402


def load_dt_column(feature_path: str, flow_suffix: str) -> np.ndarray:
    """Load flow sidecar and return D_t column (last column) as flat array.

    Args:
        feature_path: Path to the RGB .npy feature file.
        flow_suffix: Suffix used for the flow sidecar (e.g., '_flow').

    Returns:
        1-D numpy array of D_t values (one per segment).
    """
    sidecar = flow_sidecar_path(feature_path, flow_suffix=flow_suffix)
    if not Path(sidecar).exists():
        raise FileNotFoundError(f"Flow sidecar not found: {sidecar}")
    flow = np.load(sidecar, allow_pickle=True).astype(np.float32)
    if flow.ndim != 2:
        raise ValueError(f"Expected 2D flow array (T, F), got {flow.shape} for {sidecar}")
    return flow[:, -1]  # Last column = D_t


def diagnose(list_path: str, flow_suffix: str, sample_n: int, output_json: str) -> None:
    """Sample up to sample_n files, compute D_t stats, print verdict, save JSON.

    Args:
        list_path: Path to test manifest file (path label per line).
        flow_suffix: Flow sidecar suffix string.
        sample_n: Maximum number of files to sample.
        output_json: Path to save the JSON diagnostic report.
    """
    list_file = Path(list_path)
    if not list_file.exists():
        print(f"ERROR: List file not found: {list_file}", file=sys.stderr)
        sys.exit(1)

    lines = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print(f"ERROR: List file is empty: {list_file}", file=sys.stderr)
        sys.exit(1)

    # Subsample
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(lines), size=min(sample_n, len(lines)), replace=False)
    sampled = [lines[i] for i in sorted(indices)]

    all_dt: list = []
    loaded = 0
    skipped = 0

    for line in sampled:
        parts = line.split()
        if len(parts) < 2:
            skipped += 1
            continue
        feature_path = parts[0]
        try:
            dt_vals = load_dt_column(feature_path, flow_suffix)
            all_dt.extend(dt_vals.tolist())
            loaded += 1
        except FileNotFoundError as e:
            print(f"  SKIP (missing sidecar): {e}")
            skipped += 1
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}): {e}")
            skipped += 1

    if not all_dt:
        print("ERROR: No D_t values loaded. Check flow sidecar files.", file=sys.stderr)
        sys.exit(1)

    dt = np.array(all_dt, dtype=np.float64)

    # Compute statistics
    mn = float(dt.min())
    mx = float(dt.max())
    mean = float(dt.mean())
    std = float(dt.std())
    frac_pos = float((dt > 0.5).mean())
    frac_neg = float((dt < -0.5).mean())
    frac_neutral = float(((dt >= -0.2) & (dt <= 0.2)).mean())

    # Build 10-bin histogram
    hist_counts, hist_edges = np.histogram(dt, bins=10, range=(-1.0, 1.0))
    histogram = {
        "bin_edges": [round(float(e), 4) for e in hist_edges.tolist()],
        "counts": hist_counts.tolist(),
    }

    # Verdict
    verdict_lines = []
    if frac_neg < 0.02:
        verdict_lines.append(
            "WARNING: Fewer than 2% of segments have negative direction alignment. "
            "Direction signal may not distinguish wrong-way from normal."
        )
    if frac_pos > 0.95:
        verdict_lines.append(
            "WARNING: More than 95% of segments have D_t > 0.5. "
            "Distribution is heavily right-skewed — direction signal has limited discriminative power."
        )
    if not verdict_lines:
        verdict_lines.append(
            "OK: D_t distribution has reasonable spread. Direction signal appears informative."
        )

    verdict = " | ".join(verdict_lines)

    report = {
        "files_loaded": loaded,
        "files_skipped": skipped,
        "total_segments": len(all_dt),
        "dt_min": mn,
        "dt_max": mx,
        "dt_mean": mean,
        "dt_std": std,
        "fraction_above_0.5": frac_pos,
        "fraction_below_-0.5": frac_neg,
        "fraction_in_[-0.2,0.2]": frac_neutral,
        "histogram": histogram,
        "verdict": verdict,
    }

    # Print
    print(f"=== Flow D_t Diagnostic ===")
    print(f"  Files loaded   : {loaded}  (skipped: {skipped})")
    print(f"  Total segments : {len(all_dt)}")
    print(f"  D_t  min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}  std={std:.4f}")
    print(f"  Fraction > 0.5 : {frac_pos:.4f}")
    print(f"  Fraction < -0.5: {frac_neg:.4f}")
    print(f"  Fraction neutral [-0.2, 0.2]: {frac_neutral:.4f}")
    print(f"\nVerdict: {verdict}\n")

    # Save JSON
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved diagnostic report to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose D_t direction cosine column distribution in flow features"
    )
    parser.add_argument(
        "--test-list",
        default="list/splits/test_base.list",
        help="Manifest file to sample from (default: list/splits/test_base.list)",
    )
    parser.add_argument(
        "--flow-suffix",
        default="_flow",
        help="Flow sidecar suffix (default: _flow)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=40,
        help="Maximum number of files to sample (default: 40)",
    )
    parser.add_argument(
        "--output-json",
        default="results/flow_diagnostic.json",
        help="Path to save the JSON diagnostic report (default: results/flow_diagnostic.json)",
    )
    args = parser.parse_args()
    diagnose(args.test_list, args.flow_suffix, args.sample, args.output_json)


if __name__ == "__main__":
    main()
