"""Helper subpackage for DA‑RTFM.

This package contains one‑off scripts used during data preparation and experimentation.
Keeping these utilities in a dedicated subpackage makes it clear they are not
part of the core training or inference pipeline. You are free to move or
delete these helper scripts once you have generated the necessary assets.
"""

__all__ = [
    "split_dataset",
    "extract_features",
    "multi_seed_ablation",
]