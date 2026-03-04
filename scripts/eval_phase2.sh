#!/bin/bash

STEPS=300000
EVAL_STEPS=20

echo "=== Phase 2 Evaluation ==="

echo "--- Baseline ---"
python evaluate.py \
    -c configs/flowsep/flowsep.yaml \
    -l model_logs/uss/flowsep/checkpoints/last.ckpt \
    -s $EVAL_STEPS -n 200

echo "--- Exp A: Bridge ---"
python evaluate.py \
    -c configs/flowsep/bridge.yaml \
    -l model_logs/uss/bridge/checkpoints/last.ckpt \
    -s $EVAL_STEPS -n 200

echo "--- Exp B: SB Schedule ---"
python evaluate.py \
    -c configs/flowsep/sb_schedule.yaml \
    -l model_logs/uss/sb_schedule/checkpoints/last.ckpt \
    -s $EVAL_STEPS -n 200

echo "--- Exp C: Data Prediction ---"
python evaluate.py \
    -c configs/flowsep/data_pred.yaml \
    -l model_logs/uss/data_pred/checkpoints/last.ckpt \
    -s $EVAL_STEPS -n 200

echo "--- Exp D: EI Solver (on baseline ckpt) ---"
python evaluate.py \
    -c configs/flowsep/flowsep.yaml \
    -l model_logs/uss/flowsep/checkpoints/last.ckpt \
    -s 5 -n 200
python evaluate.py \
    -c configs/flowsep/flowsep.yaml \
    -l model_logs/uss/flowsep/checkpoints/last.ckpt \
    -s 10 -n 200
python evaluate.py \
    -c configs/flowsep/flowsep.yaml \
    -l model_logs/uss/flowsep/checkpoints/last.ckpt \
    -s 50 -n 200

echo "=== Phase 2 Evaluation Complete ==="
