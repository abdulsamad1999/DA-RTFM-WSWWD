#!/bin/bash
# Run all baseline methods.
# Run from the DA-RTFM-release directory.
set -eo pipefail
mkdir -p logs results

TRAIN="list/splits/train_grouped_v3.list"
VAL="list/splits/val_base_v3.list"
TEST="list/splits/test_base_v3.list"

echo "=== Flow Direction Baseline ==="
python baselines/flow_direction_baseline.py \
    --test-list "$TEST" --val-list "$VAL" \
    --use-flow --num-segments 32 --precision-floor 0.9 \
    --output-json results/flow_direction_baseline.json \
    2>&1 | tee logs/flow_direction_baseline.log

echo "=== Logistic Regression Baseline ==="
python baselines/logistic_regression_baseline.py \
    --train-list "$TRAIN" --val-list "$VAL" --test-list "$TEST" \
    --use-flow --num-segments 32 --precision-floor 0.9 \
    --output-json results/lr_baseline.json \
    2>&1 | tee logs/lr_baseline.log

echo "=== Supervised Baseline (3 seeds) ==="
for SEED in 42 1337 2026; do
    python baselines/supervised_baseline.py \
        --train-list "$TRAIN" --val-list "$VAL" --test-list "$TEST" \
        --seed "$SEED" --num-segments 32 --precision-floor 0.9 \
        --output-json "results/supervised_baseline_seed${SEED}.json" \
        2>&1 | tee "logs/supervised_seed${SEED}.log"
done

echo "=== YOLO+Tracking Baseline (requires raw videos + ultralytics) ==="
python baselines/yolo_tracking_baseline.py \
    --test-list "$TEST" --val-list "$VAL" --train-list "$TRAIN" \
    --video-root dataset \
    --precision-floor 0.9 \
    --output-json results/yolo_tracking_baseline.json \
    2>&1 | tee logs/yolo_tracking_baseline.log \
    || echo "YOLO baseline skipped (check logs/yolo_tracking_baseline.log)"

echo "All baselines complete."