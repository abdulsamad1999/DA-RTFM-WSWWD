@echo off
setlocal enabledelayedexpansion

set TRAIN=list\splits\train_grouped_v3.list
set VAL=list\splits\val_base_v3.list
set TEST=list\splits\test_base_v3.list

if not exist logs   mkdir logs
if not exist results mkdir results

echo === Flow Direction Baseline ===
python baselines\flow_direction_baseline.py ^
    --test-list %TEST% --val-list %VAL% ^
    --use-flow --num-segments 32 --precision-floor 0.9 ^
    --output-json results\flow_direction_baseline.json ^
    > logs\flow_direction_baseline.log 2>&1

echo === Logistic Regression Baseline ===
python baselines\logistic_regression_baseline.py ^
    --train-list %TRAIN% --val-list %VAL% --test-list %TEST% ^
    --use-flow --num-segments 32 --precision-floor 0.9 ^
    --output-json results\lr_baseline.json ^
    > logs\lr_baseline.log 2>&1

echo === Supervised Baseline (3 seeds) ===
for %%S in (42 1337 2026) do (
    python baselines\supervised_baseline.py ^
        --train-list %TRAIN% --val-list %VAL% --test-list %TEST% ^
        --seed %%S --num-segments 32 --precision-floor 0.9 ^
        --output-json results\supervised_baseline_seed%%S.json ^
        > logs\supervised_seed%%S.log 2>&1
)

echo === YOLO+Tracking Baseline (requires raw videos + ultralytics) ===
python baselines\yolo_tracking_baseline.py ^
    --test-list %TEST% --val-list %VAL% --train-list %TRAIN% ^
    --video-root dataset ^
    --precision-floor 0.9 ^
    --output-json results\yolo_tracking_baseline.json ^
    > logs\yolo_tracking_baseline.log 2>&1

echo All baselines complete.