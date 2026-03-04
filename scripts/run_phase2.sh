#!/bin/bash

CUDA_A="0,1"
CUDA_B="2,3"
CUDA_C="4,5"
CUDA_D="6,7"

echo "=== Phase 2: Ablation Experiments ==="
echo "Exp A (Bridge):        GPU $CUDA_A"
echo "Exp B (SB Schedule):   GPU $CUDA_B"
echo "Exp C (Data Pred):     GPU $CUDA_C"
echo "Baseline (continue):   GPU $CUDA_D"

CUDA_VISIBLE_DEVICES=$CUDA_A python train.py -c configs/flowsep/bridge.yaml &
PID_A=$!

CUDA_VISIBLE_DEVICES=$CUDA_B python train.py -c configs/flowsep/sb_schedule.yaml &
PID_B=$!

CUDA_VISIBLE_DEVICES=$CUDA_C python train.py -c configs/flowsep/data_pred.yaml &
PID_C=$!

CUDA_VISIBLE_DEVICES=$CUDA_D python train.py -c configs/flowsep/flowsep.yaml &
PID_D=$!

echo "PIDs: A=$PID_A B=$PID_B C=$PID_C D=$PID_D"
echo "Waiting for all experiments to finish..."
wait
echo "All Phase 2 experiments completed."
