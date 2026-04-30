import argparse

parser = argparse.ArgumentParser(description='RTFM (Wrong-way driving)')

# Core arguments
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--modality', default='RGB', help='Input modality (RGB, AUDIO, MIX)')
parser.add_argument('--gt', default='list/gt-wrongway.npy', help='Optional ground-truth .npy file for testing')
parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use (kept for compatibility)')
parser.add_argument('--lr', type=str, default='[0.0001]*15000', help='Learning-rate schedule as a python list string')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for each of normal/anomaly loaders')
parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers')
parser.add_argument('--model-name', default='rtfm_nosparse', help='Checkpoint/model name prefix')
parser.add_argument('--pretrained-ckpt', default=None, help='Path to pretrained checkpoint (optional)')
parser.add_argument('--num-classes', type=int, default=1, help='Number of output classes (binary=1)')
parser.add_argument('--dataset', default='wrongway-dataset', help='Dataset name (e.g., wrongway-dataset)')
parser.add_argument('--plot-freq', type=int, default=10, help='Logging frequency')
parser.add_argument('--max-epoch', type=int, default=100, help='Number of training epochs')

# Feature lists (each line: "<path_to_feature.npy> <label>")
parser.add_argument('--rgb-list', default='list/mytrain.list', help='Training list file')
parser.add_argument('--test-rgb-list', default='list/mytest.list', help='Testing list file')

# Feature shape
parser.add_argument('--feature-size', type=int, default=1024, help='RGB feature dimension per segment')
parser.add_argument('--num-segments', type=int, default=32, help='Number of temporal segments per video')

# Flow feature fusion
parser.add_argument('--use-flow', action='store_true',
                    help='Concatenate per-segment optical-flow features (u,v,mag,D) to RGB features')
parser.add_argument('--flow-suffix', type=str, default='_flow',
                    help='Suffix for flow feature files: <rgb_path_wo_ext><suffix>.npy')
parser.add_argument('--flow-dim', type=int, default=4,
                    help='Flow feature dimension per segment (default 4 = [u,v,mag,D])')

# Temporal smoothness regularization (feature-level)
parser.add_argument('--lambda-temp', type=float, default=0.0,
                    help='Weight for feature-level temporal smoothness regularization')
parser.add_argument('--temp-on-all', action='store_true',
                    help='If set, apply temporal smoothness to both normal and abnormal; otherwise normal-only')

# Direction-opposition penalty (normal-only, per supervisor wording)
parser.add_argument('--lambda-dir', type=float, default=0.0,
                    help='Weight for direction-opposition penalty on NORMAL videos only')
parser.add_argument('--dir-margin', type=float, default=0.0,
                    help='Margin for direction consistency D_t; penalize when D_t < margin (0 penalizes negative alignment)')

# Optional legacy score smoothness
parser.add_argument('--lambda-score-smooth', type=float, default=0.0,
                    help='Optional weight for score-level smoothness regularization (legacy)')

# List splitting for custom datasets
parser.add_argument('--split-by-label', action='store_true',
                    help='Split train list into normal/anomaly loaders using the label column (required for GTA5 lists)')

# Disable visdom
# Visualization has been completely removed from the codebase; this flag remains for backwards compatibility but has no effect.
parser.add_argument('--no-viz', action='store_true', help='(Deprecated) Visdom support has been removed; this flag is ignored')


# Reproducibility
parser.add_argument('--seed', type=int, default=123, help='Random seed for numpy/torch/python')
parser.add_argument('--deterministic', action='store_true', help='Enable deterministic CuDNN (may reduce speed)')

# Output management
parser.add_argument(
    '--output-root',
    type=str,
    default='runs',
    help='Root directory under which timestamped run directories will be created. '
         'During training a new subdirectory will be made within this root to hold '
         'checkpoints, logs and metrics.'
)

# Visdom has been removed; environment/server/port options are no longer used.


# Temporal smoothness normalization
parser.add_argument('--temp-norm-dim', action='store_true',
                    help='Also normalize temporal smoothness by feature dimension D (recommended)')

# ---------------------------------------------------------------------------
# Validation / splitting options
#
# Users can either provide a separate validation list via --val-rgb-list or
# specify a fraction of the training list to reserve for validation via
# --val-ratio. If both are omitted, no validation is performed (not
# recommended). Validation metrics are used to select the best checkpoint
# and operating threshold without peeking at the test split.
parser.add_argument(
    '--val-rgb-list', type=str, default=None,
    help=(
        'Optional validation list file. If provided, this list should contain "<path> <label>" '
        'lines like the training list and will be used exclusively for validation. When omitted, '
        'a portion of the training list will be held out for validation according to --val-ratio.'
    ),
)
parser.add_argument(
    '--val-ratio', type=float, default=0.2,
    help=(
        'Fraction of the training list to reserve as a validation set when --val-rgb-list is not provided. '
        'Must be between 0 and 1.'
    ),
)

# Top‑k magnitude ranking ratio. The base model selects the top k segments
# according to feature magnitude for positive/negative bag ranking. The
# value of k is derived as k = max(1, int(topk_ratio * T)), where T is
# --num-segments. Setting this explicitly ensures consistency with the
# paper, which uses k=8 for T=32 (topk_ratio=0.25).
parser.add_argument(
    '--label-smooth-eps', type=float, default=0.0,
    help=(
        'Label smoothing epsilon for BCE classification loss. '
        '0.1 prevents sigmoid saturation from magnitude ranking pressure '
        '(positive label -> 0.9, negative label -> 0.1). '
        '0.0 = disabled (original behaviour).'
    ),
)

parser.add_argument(
    '--topk-ratio', type=float, default=0.25,
    help=(
        'Ratio of segments used for top‑k magnitude ranking. For T segments, k = max(1, int(topk_ratio * T)). '
        'Defaults to 0.25 (use top 25%% of segments).'
    ),
)
