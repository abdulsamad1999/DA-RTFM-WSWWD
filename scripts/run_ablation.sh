#!/bin/bash
# Ablation study: Conditions A/B/C, 3 seeds each.
# Run from the DA-RTFM-release directory.
set -eo pipefail
mkdir -p logs results

TRAIN="list/splits/train_grouped_v3.list"
VAL="list/splits/val_base_v3.list"
TEST="list/splits/test_base_v3.list"
OUT="runs_ablation"
FLOOR=0.85

BASE_ARGS="--rgb-list $TRAIN --val-rgb-list $VAL --test-rgb-list $TEST \
           --dataset wrongway-dataset --num-segments 32 --topk-ratio 0.25 \
           --workers 0 --output-root $OUT --label-smooth-eps 0.1"

declare -A COND_ARGS
COND_ARGS[A]="--model-name ablation_A"
COND_ARGS[B]="--model-name ablation_B --use-flow"
COND_ARGS[C]="--model-name ablation_C --use-flow --lambda-temp 1e-4 --temp-on-all"

for COND in A B C; do
    EXTRA="${COND_ARGS[$COND]}"
    MODEL_NAME="ablation_${COND}"
    FLOW_ARGS=""
    if [[ "$EXTRA" == *"--use-flow"* ]]; then FLOW_ARGS="--use-flow"; fi

    for SEED in 42 1337 2026; do
        echo "=== Ablation Condition $COND  seed=$SEED ==="

        python main.py $BASE_ARGS $EXTRA --seed "$SEED" \
            2>&1 | tee "logs/ablation_${COND}_seed${SEED}_train.log"

        RUN_DIR=$(ls -td ${OUT}/*_${MODEL_NAME}/ 2>/dev/null | head -1)
        CKPT="${RUN_DIR}ckpt/${MODEL_NAME}_best.pkl"
        CAL="${RUN_DIR}metrics/calibration_report.json"

        python calibrate_thresholds.py \
            --model-path "$CKPT" \
            --rgb-list "$TRAIN" --val-rgb-list "$VAL" --test-rgb-list "$TEST" \
            --dataset wrongway-dataset \
            --num-segments 32 --topk-ratio 0.25 \
            $FLOW_ARGS --workers 0 \
            --precision-floor "$FLOOR" --alert-rule video_score \
            --output-report "$CAL" \
            2>&1 | tee "logs/ablation_${COND}_seed${SEED}_calibration.log"

        THRESHOLD=$(python -c "import json; d=json.load(open('${CAL}')); print(d['validation']['SelectedRule']['thr'])")
        mkdir -p "${RUN_DIR}reports"

        python helpers/evaluate_fixed_threshold.py \
            --model-path "$CKPT" \
            --rgb-list "$TRAIN" --test-rgb-list "$TEST" \
            --dataset wrongway-dataset \
            --num-segments 32 --topk-ratio 0.25 \
            $FLOW_ARGS --workers 0 \
            --threshold "$THRESHOLD" \
            --split-name "ablation_${COND}_seed${SEED}" \
            --report-json "${RUN_DIR}reports/frozen_test_report.json" \
            --metrics-text "${RUN_DIR}metrics/frozen_test_metrics.txt" \
            2>&1 | tee "logs/ablation_${COND}_seed${SEED}_frozen.log"
    done

    python helpers/compute_run_stats.py --root "$OUT" --filter "$MODEL_NAME" \
        2>&1 | tee "results/ablation_${COND}_aggregated.txt"
done

echo "All ablation runs complete."