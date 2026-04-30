@echo off
setlocal enabledelayedexpansion

set TRAIN=list\splits\train_grouped_v3.list
set VAL=list\splits\val_base_v3.list
set TEST=list\splits\test_base_v3.list
set OUT=runs_da_rtfm
set FLOOR=0.85

if not exist logs   mkdir logs
if not exist results mkdir results

for %%S in (42 1337 2026) do (
    echo === DA-RTFM seed=%%S ===

    python main.py ^
        --rgb-list %TRAIN% ^
        --val-rgb-list %VAL% ^
        --test-rgb-list %TEST% ^
        --model-name rtfm_wd ^
        --dataset wrongway-dataset ^
        --num-segments 32 --topk-ratio 0.25 ^
        --use-flow --workers 0 ^
        --lambda-temp 1e-4 --temp-on-all ^
        --lambda-dir 1e-2 --dir-margin 0.0 ^
        --label-smooth-eps 0.1 ^
        --seed %%S --output-root %OUT% ^
        > logs\da_rtfm_seed%%S_train.log 2>&1

    for /f "delims=" %%D in ('python -c "from pathlib import Path; runs=sorted(Path(r'%OUT%').glob('*_rtfm_wd')); print(str(runs[-1])+chr(92))"') do set RUN_DIR=%%D

    python calibrate_thresholds.py ^
        --model-path !RUN_DIR!ckpt\rtfm_wd_best.pkl ^
        --rgb-list %TRAIN% --val-rgb-list %VAL% --test-rgb-list %TEST% ^
        --dataset wrongway-dataset ^
        --num-segments 32 --topk-ratio 0.25 ^
        --use-flow --workers 0 ^
        --precision-floor %FLOOR% --alert-rule video_score ^
        --output-report !RUN_DIR!metrics\calibration_report.json ^
        > logs\da_rtfm_seed%%S_calibration.log 2>&1

    for /f "delims=" %%T in ('python -c "import json; d=json.load(open(r'!RUN_DIR!metrics\calibration_report.json')); print(d['validation']['SelectedRule']['thr'])"') do set THRESHOLD=%%T

    if not exist !RUN_DIR!reports mkdir !RUN_DIR!reports

    python helpers\evaluate_fixed_threshold.py ^
        --model-path !RUN_DIR!ckpt\rtfm_wd_best.pkl ^
        --rgb-list %TRAIN% --test-rgb-list %TEST% ^
        --dataset wrongway-dataset ^
        --num-segments 32 --topk-ratio 0.25 ^
        --use-flow --workers 0 ^
        --threshold !THRESHOLD! ^
        --split-name da_rtfm_seed%%S ^
        --report-json !RUN_DIR!reports\frozen_test_report.json ^
        --metrics-text !RUN_DIR!metrics\frozen_test_metrics.txt ^
        > logs\da_rtfm_seed%%S_frozen.log 2>&1

    echo === Seed %%S done ===
)

python helpers\compute_run_stats.py --root %OUT% --filter rtfm_wd > results\da_rtfm_aggregated.txt 2>&1
echo All DA-RTFM runs complete.