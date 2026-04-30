@echo off
setlocal enabledelayedexpansion

set TRAIN=list\splits\train_grouped_v3.list
set VAL=list\splits\val_base_v3.list
set TEST=list\splits\test_base_v3.list
set OUT=runs_ablation
set FLOOR=0.85

if not exist logs   mkdir logs
if not exist results mkdir results

for %%C in (A B C) do (
    if "%%C"=="A" set EXTRA=--model-name ablation_A
    if "%%C"=="B" set EXTRA=--model-name ablation_B --use-flow
    if "%%C"=="C" set EXTRA=--model-name ablation_C --use-flow --lambda-temp 1e-4 --temp-on-all

    set MODEL_NAME=ablation_%%C

    for %%S in (42 1337 2026) do (
        echo === Ablation Condition %%C  seed=%%S ===

        python main.py ^
            --rgb-list %TRAIN% --val-rgb-list %VAL% --test-rgb-list %TEST% ^
            --dataset wrongway-dataset --num-segments 32 --topk-ratio 0.25 ^
            --workers 0 --output-root %OUT% --label-smooth-eps 0.1 ^
            !EXTRA! --seed %%S ^
            > logs\ablation_%%C_seed%%S_train.log 2>&1

        for /f "delims=" %%D in ('python -c "from pathlib import Path; runs=sorted(Path(r'%OUT%').glob('*_ablation_%%C')); print(str(runs[-1])+chr(92))"') do set RUN_DIR=%%D

        set FLOW_OPT=
        if "%%C"=="B" set FLOW_OPT=--use-flow
        if "%%C"=="C" set FLOW_OPT=--use-flow

        python calibrate_thresholds.py ^
            --model-path !RUN_DIR!ckpt\ablation_%%C_best.pkl ^
            --rgb-list %TRAIN% --val-rgb-list %VAL% --test-rgb-list %TEST% ^
            --dataset wrongway-dataset --num-segments 32 --topk-ratio 0.25 ^
            !FLOW_OPT! --workers 0 ^
            --precision-floor %FLOOR% --alert-rule video_score ^
            --output-report !RUN_DIR!metrics\calibration_report.json ^
            > logs\ablation_%%C_seed%%S_calibration.log 2>&1

        for /f "delims=" %%T in ('python -c "import json; d=json.load(open(r'!RUN_DIR!metrics\calibration_report.json')); print(d['validation']['SelectedRule']['thr'])"') do set THRESHOLD=%%T

        if not exist !RUN_DIR!reports mkdir !RUN_DIR!reports

        python helpers\evaluate_fixed_threshold.py ^
            --model-path !RUN_DIR!ckpt\ablation_%%C_best.pkl ^
            --rgb-list %TRAIN% --test-rgb-list %TEST% ^
            --dataset wrongway-dataset --num-segments 32 --topk-ratio 0.25 ^
            !FLOW_OPT! --workers 0 ^
            --threshold !THRESHOLD! ^
            --split-name ablation_%%C_seed%%S ^
            --report-json !RUN_DIR!reports\frozen_test_report.json ^
            --metrics-text !RUN_DIR!metrics\frozen_test_metrics.txt ^
            > logs\ablation_%%C_seed%%S_frozen.log 2>&1
    )

    python helpers\compute_run_stats.py --root %OUT% --filter ablation_%%C > results\ablation_%%C_aggregated.txt 2>&1
)

echo All ablation runs complete.