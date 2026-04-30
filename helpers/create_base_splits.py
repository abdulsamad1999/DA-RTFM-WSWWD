#!/usr/bin/env python3
"""Filter existing split lists to contain only base scenes (no _flip or _gray variants).

Creates val_base and test_base lists from grouped/full lists by dropping
augmented variants (_flip, _gray).  Training lists are NOT modified.

Usage (new, with explicit paths):
    python helpers/create_base_splits.py \
        --val-input  list/splits_v2/val_grouped.list \
        --val-output list/splits_v2/val_base_v2.list \
        --test-input list/splits_v2/test_all.list \
        --test-output list/splits_v2/test_base_v2.list

Legacy (no args — uses original hardcoded paths):
    python helpers/create_base_splits.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))


def is_base_scene(line: str) -> bool:
    """Return True if the feature path is a base scene (no _flip or _gray suffix before extension)."""
    parts = line.strip().split()
    if not parts:
        return False
    path = Path(parts[0])
    stem = path.stem  # e.g. Wrongway129_x264 or Wrongway129_flip_x264
    return "_flip" not in stem and "_gray" not in stem


def filter_base_only(input_path: Path, output_path: Path) -> tuple:
    """Filter list to base scenes only. Returns (original_count, base_count, normal, anomaly)."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    lines = [ln for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    original_count = len(lines)

    base_lines = [ln for ln in lines if is_base_scene(ln)]

    normal_count = sum(1 for ln in base_lines if ln.strip().split()[-1] == "0")
    anomaly_count = sum(1 for ln in base_lines if ln.strip().split()[-1] == "1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(base_lines) + "\n", encoding="utf-8")

    return original_count, len(base_lines), normal_count, anomaly_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter split lists to base scenes only")
    parser.add_argument("--val-input",   default=None,
                        help="Input validation list (grouped, with augmentations)")
    parser.add_argument("--val-output",  default=None,
                        help="Output validation list (base scenes only)")
    parser.add_argument("--test-input",  default=None,
                        help="Input test list")
    parser.add_argument("--test-output", default=None,
                        help="Output test list (base scenes only)")
    args = parser.parse_args()

    # Fall back to legacy hardcoded paths when no args given
    splits_dir = REPO_ROOT / "list" / "splits"
    val_in  = Path(args.val_input)  if args.val_input  else splits_dir / "val_grouped.list"
    val_out = Path(args.val_output) if args.val_output else splits_dir / "val_base.list"
    test_in  = Path(args.test_input)  if args.test_input  else splits_dir / "test.list"
    test_out = Path(args.test_output) if args.test_output else splits_dir / "test_base.list"

    # val -> val_base
    orig, base, normal, anomaly = filter_base_only(val_in, val_out)
    print(f"{val_out.name}:")
    print(f"  Original entries : {orig}")
    print(f"  Base-only entries: {base} (normal={normal}, anomaly={anomaly})")
    print(f"  Written to       : {val_out}")

    # test -> test_base
    orig, base, normal, anomaly = filter_base_only(test_in, test_out)
    print(f"\n{test_out.name}:")
    print(f"  Original entries : {orig}")
    print(f"  Base-only entries: {base} (normal={normal}, anomaly={anomaly})")
    print(f"  Written to       : {test_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
