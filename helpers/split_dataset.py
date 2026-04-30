#!/usr/bin/env python3
"""Dataset list splitter for DA-RTFM."""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from feature_utils import base_scene_id  # noqa: E402


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in (line.strip() for line in f) if ln]


def write_list(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def parse_label(line: str) -> int:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid manifest line: {line}")
    return int(float(parts[-1]))


def split_by_file(lines: List[str], val_ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    groups: Dict[int, List[str]] = defaultdict(list)
    for line in lines:
        groups[parse_label(line)].append(line)

    train_lines: List[str] = []
    val_lines: List[str] = []
    for label, label_lines in groups.items():
        rng.shuffle(label_lines)
        n_val = max(1, int(round(len(label_lines) * val_ratio))) if label_lines else 0
        val_lines.extend(label_lines[:n_val])
        train_lines.extend(label_lines[n_val:])
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines


def split_by_scene(lines: List[str], val_ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    scene_groups: Dict[int, List[Tuple[str, List[str]]]] = defaultdict(list)
    temp: Dict[str, List[str]] = defaultdict(list)
    scene_labels: Dict[str, int] = {}

    for line in lines:
        label = parse_label(line)
        scene = base_scene_id(line.split()[0])
        if scene in scene_labels and scene_labels[scene] != label:
            raise ValueError(f"Inconsistent labels for base scene {scene}")
        scene_labels[scene] = label
        temp[scene].append(line)

    for scene, scene_lines in temp.items():
        scene_groups[scene_labels[scene]].append((scene, scene_lines))

    train_lines: List[str] = []
    val_lines: List[str] = []
    for label, grouped in scene_groups.items():
        rng.shuffle(grouped)
        n_val = max(1, int(round(len(grouped) * val_ratio))) if grouped else 0
        val_groups = grouped[:n_val]
        train_groups = grouped[n_val:]
        for _, scene_lines in train_groups:
            train_lines.extend(scene_lines)
        for _, scene_lines in val_groups:
            val_lines.extend(scene_lines)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines


def summarize(lines: List[str]) -> dict:
    summary = {
        "files": len(lines),
        "class_counts": {0: 0, 1: 0},
        "scenes": set(),
        "scene_class_counts": {0: set(), 1: set()},
    }
    for line in lines:
        label = parse_label(line)
        scene = base_scene_id(line.split()[0])
        summary["class_counts"][label] = summary["class_counts"].get(label, 0) + 1
        summary["scenes"].add(scene)
        summary["scene_class_counts"].setdefault(label, set()).add(scene)
    return summary


def print_summary(name: str, lines: List[str]) -> None:
    summary = summarize(lines)
    print(
        f"{name}: files={summary['files']} "
        f"(normal={summary['class_counts'].get(0, 0)}, anomaly={summary['class_counts'].get(1, 0)})"
    )
    print(
        f"{name}: base_scenes={len(summary['scenes'])} "
        f"(normal={len(summary['scene_class_counts'].get(0, set()))}, "
        f"anomaly={len(summary['scene_class_counts'].get(1, set()))})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset list into training and validation sets")
    parser.add_argument("--input", required=True, help="Path to the original list file (path label)")
    parser.add_argument("--output-dir", required=True, help="Directory to write the split lists")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible splits")
    parser.add_argument("--group-by-base-scene", action="store_true", help="Keep _flip/_gray/original variants of a scene in one split")
    parser.add_argument("--train-name", default="train.list", help="Output filename for the training split")
    parser.add_argument("--val-name", default="val.list", help="Output filename for the validation split")
    args = parser.parse_args()

    lines = read_list(args.input)
    rng = random.Random(args.seed)
    if args.group_by_base_scene:
        train_lines, val_lines = split_by_scene(lines, args.val_ratio, rng)
    else:
        train_lines, val_lines = split_by_file(lines, args.val_ratio, rng)

    train_path = os.path.join(args.output_dir, args.train_name)
    val_path = os.path.join(args.output_dir, args.val_name)
    write_list(train_path, train_lines)
    write_list(val_path, val_lines)

    print_summary("train", train_lines)
    print_summary("val", val_lines)
    train_scenes = summarize(train_lines)["scenes"]
    val_scenes = summarize(val_lines)["scenes"]
    print(f"base_scene_overlap={len(train_scenes & val_scenes)}")
    print(f"Wrote {len(train_lines)} training samples to {train_path}")
    print(f"Wrote {len(val_lines)} validation samples to {val_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
