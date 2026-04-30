import os
from pathlib import Path
import re
from typing import Iterable, List, Tuple

import numpy as np


def read_manifest(list_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                raise ValueError(f"Invalid manifest line: {line.strip()}")
            items.append((parts[0], int(float(parts[1]))))
    return items


def adaptive_pool(feats: np.ndarray, length: int) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"adaptive_pool expects a 2D array, got {feats.shape}")
    total = feats.shape[0]
    if total == 0:
        raise ValueError("Cannot pool an empty feature array")
    if total == length:
        return feats.astype(np.float32, copy=False)
    bins = np.linspace(0, total, length + 1, dtype=int)
    pooled = np.zeros((length, feats.shape[1]), dtype=np.float32)
    for idx in range(length):
        start, end = bins[idx], bins[idx + 1]
        pooled[idx] = feats[start:end].mean(axis=0) if start < end else feats[start]
    return pooled


def normalize_manifest_path(path: str) -> str:
    return Path(path).as_posix()


def infer_class_name(feature_path: str, label: int) -> str:
    parts = [part.lower() for part in Path(feature_path).parts]
    if "normal" in parts:
        return "Normal"
    if "anomaly" in parts:
        return "Anomaly"
    return "Anomaly" if int(label) == 1 else "Normal"


def base_scene_id(feature_path: str) -> str:
    stem = Path(feature_path).stem
    stem = re.sub(r"(_flip|_gray)(?=_x264$)", "", stem)
    return stem


def rewrite_feature_path(feature_path: str, label: int, feature_root: str) -> str:
    class_name = infer_class_name(feature_path, label)
    filename = Path(feature_path).name
    return normalize_manifest_path(str(Path(feature_root) / class_name / filename))


def resolve_manifest_path(path: str, base_dir: str = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return candidate.resolve()


def flow_sidecar_path(feature_path: str, flow_suffix: str = "_flow") -> str:
    path = Path(feature_path)
    return normalize_manifest_path(str(path.with_name(f"{path.stem}{flow_suffix}{path.suffix}")))


def load_feature_array(
    feature_path: str,
    num_segments: int = None,
    use_flow: bool = False,
    flow_suffix: str = "_flow",
) -> np.ndarray:
    rgb = np.load(feature_path, allow_pickle=True).astype(np.float32)
    if num_segments is not None:
        rgb = adaptive_pool(rgb, num_segments)
    if not use_flow:
        return rgb
    flow_path = flow_sidecar_path(feature_path, flow_suffix=flow_suffix)
    flow = np.load(flow_path, allow_pickle=True).astype(np.float32)
    if num_segments is not None:
        flow = adaptive_pool(flow, num_segments)
    if flow.shape[0] != rgb.shape[0]:
        raise ValueError(
            f"RGB/flow segment mismatch for {feature_path}: {rgb.shape[0]} vs {flow.shape[0]}"
        )
    return np.concatenate([rgb, flow], axis=1).astype(np.float32)


def manifest_class_counts(items: Iterable[Tuple[str, int]]) -> dict:
    counts = {0: 0, 1: 0}
    for _, label in items:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts
