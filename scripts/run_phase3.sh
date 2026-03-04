#!/bin/bash

CUDA_DEVICES="0,1,2,3"

echo "=== Phase 3: Full Training with Best Combination ==="
echo "Config: bridge + SB schedule + data prediction + EI solver"
echo "GPUs: $CUDA_DEVICES"
echo "Max steps: 2,000,000"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python train.py -c configs/flowsep/bridge_sb.yaml
