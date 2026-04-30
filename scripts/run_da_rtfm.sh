#!/bin/bash
# Train DA-RTFM (3 seeds), calibrate thresholds, evaluate.
# Run from the DA-RTFM-release directory.
set -eo pipefail
mkdir -p logs results

TRAIN="list/splits/train_grouped_v3.list"
VAL="list/splits/val_base_v3.list"
TEST="list/splits/test_base_v3.list"
OUT="runs_da_rtfm"
FLOOR=0.85

for SEED in 42 1337 2026; do
    echo "=== DA-RTFM seed=$SEED ==="

    python main.py \
        --rgb-list "$TRAIN" \
        --val-rgb-list "$VAL" \
        --test-rgb-list "$TEST" \
        --model-name rtfm_wd \
        --dataset wrongway-dataset \
        --num-segments 32 --topk-ratio 0.25 \
        --use-flow --workers 0 \
        --lambda-temp 1e-4 --temp-on-all \
        --lambda-dir 1e-2 --dir-margin 0.0 \
        --label-smooth-eps 0.1 \
        --seed "$SEED" --output-root "$OUT" \
        2>&1 | tee "logs/da_rtfm_seed${SEED}_train.log"

    RUN_DIR=$(ls -td ${OUT}/*_rtfm_wd/ 2>/dev/null | head -1)
    CKPT="${RUN_DIR}ckpt/rtfm_wd_best.pkl"
    CAL="${RUN_DIR}metrics/calibration_report.json"

    python calibrate_thresholds.py \
        --model-path "$CKPT" \
        --rgb-list "$TRAIN" --val-rgb-list "$VAL" --test-rgb-list "$TEST" \
        --dataset wrongway-dataset \
        --num-segments 32 --topk-ratio 0.25 \
        --use-flow --workers 0 \
        --precision-floor "$FLOOR" --alert-rule video_score \
        --output-report "$CAL" \
        2>&1 | tee "logs/da_rtfm_seed${SEED}_calibration.log"

    THRESHOLD=$(python -c "import json; d=json.load(open('${CAL}')); print(d['validation']['SelectedRule']['thr'])")
    mkdir -p "${RUN_DIR}reports"

    python helpers/evaluate_fixed_threshold.py \
        --model-path "$CKPT" \
        --rgb-list "$TRAIN" --test-rgb-list "$TEST" \
        --dataset wrongway-dataset \
        --num-segments 32 --topk-ratio 0.25 \
        --use-flow --workers 0 \
        --threshold "$THRESHOLD" \
        --split-name "da_rtfm_seed${SEED}" \
        --report-json "${RUN_DIR}reports/frozen_test_report.json" \
        --metrics-text "${RUN_DIR}metrics/frozen_test_metrics.txt" \
        2>&1 | tee "logs/da_rtfm_seed${SEED}_frozen.log"

    echo "=== Seed $SEED done ==="
done

python helpers/compute_run_stats.py --root "$OUT" --filter rtfm_wd \
    2>&1 | tee results/da_rtfm_aggregated.txt
echo "All DA-RTFM runs complete."