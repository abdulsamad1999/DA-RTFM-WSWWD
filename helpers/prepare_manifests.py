#!/usr/bin/env python3
"""Rewrite, generate, and verify DA-RTFM feature manifests."""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from feature_utils import (  # noqa: E402
    flow_sidecar_path,
    manifest_class_counts,
    read_manifest,
    rewrite_feature_path,
)


def write_manifest(path: str, items: Iterable[Tuple[str, int]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for feature_path, label in items:
            f.write(f"{feature_path} {int(label)}\n")


def rewrite_manifest(input_path: str, output_path: str, feature_root: str) -> None:
    items = read_manifest(input_path)
    rewritten = [
        (rewrite_feature_path(feature_path, label, feature_root), label)
        for feature_path, label in items
    ]
    write_manifest(output_path, rewritten)
    counts = manifest_class_counts(rewritten)
    print(f"[OK] Rewrote {len(rewritten)} entries to {output_path}")
    print(f"      Class counts: normal={counts.get(0, 0)}, anomaly={counts.get(1, 0)}")


def generate_manifest_from_videos(video_root: str, output_path: str, feature_root: str) -> None:
    root = Path(video_root)
    items: List[Tuple[str, int]] = []
    for class_name, label in (("Normal", 0), ("Anomaly", 1)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for video_path in sorted(class_dir.rglob("*.mp4")):
            feature_name = f"{video_path.stem}.npy"
            feature_path = Path(feature_root) / class_name / feature_name
            items.append((feature_path.as_posix(), label))
    write_manifest(output_path, items)
    counts = manifest_class_counts(items)
    print(f"[OK] Generated {len(items)} entries at {output_path}")
    print(f"      Class counts: normal={counts.get(0, 0)}, anomaly={counts.get(1, 0)}")


def verify_manifest(list_path: str, with_flow: bool, flow_suffix: str) -> int:
    items = read_manifest(list_path)
    missing_rgb = []
    missing_flow = []
    for feature_path, _ in items:
        if not Path(feature_path).exists():
            missing_rgb.append(feature_path)
            continue
        if with_flow:
            sidecar = flow_sidecar_path(feature_path, flow_suffix=flow_suffix)
            if not Path(sidecar).exists():
                missing_flow.append(sidecar)
    print(
        f"[VERIFY] {list_path}: entries={len(items)}, missing_rgb={len(missing_rgb)}, "
        f"missing_flow={len(missing_flow)}"
    )
    if missing_rgb:
        print(f"          Missing RGB sample: {missing_rgb[:3]}")
    if missing_flow:
        print(f"          Missing flow sample: {missing_flow[:3]}")
    return 1 if (missing_rgb or missing_flow) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DA-RTFM manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite_parser = subparsers.add_parser("rewrite", help="Rewrite a manifest to a new feature root")
    rewrite_parser.add_argument("--input", required=True, help="Input manifest path")
    rewrite_parser.add_argument("--output", required=True, help="Output manifest path")
    rewrite_parser.add_argument("--feature-root", required=True, help="Feature root to rewrite into")

    generate_parser = subparsers.add_parser("generate", help="Generate a manifest from a video directory")
    generate_parser.add_argument("--video-root", required=True, help="Video directory with Normal/Anomaly subfolders")
    generate_parser.add_argument("--output", required=True, help="Output manifest path")
    generate_parser.add_argument("--feature-root", required=True, help="Feature root referenced by the generated manifest")

    verify_parser = subparsers.add_parser("verify", help="Verify feature paths referenced by a manifest")
    verify_parser.add_argument("--list", required=True, help="Manifest to verify")
    verify_parser.add_argument("--with-flow", action="store_true", help="Require *_flow.npy sidecars")
    verify_parser.add_argument("--flow-suffix", default="_flow", help="Sidecar suffix used for flow features")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "rewrite":
        rewrite_manifest(args.input, args.output, args.feature_root)
    elif args.command == "generate":
        generate_manifest_from_videos(args.video_root, args.output, args.feature_root)
    elif args.command == "verify":
        raise SystemExit(verify_manifest(args.list, args.with_flow, args.flow_suffix))


if __name__ == "__main__":  # pragma: no cover
    main()
