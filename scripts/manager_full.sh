#!/bin/bash

echo "Running Manager Agent Full Dataset Evaluation..."

# Set GPU
export CUDA_VISIBLE_DEVICES=1


# Run evaluation with full dataset (0 samples = all samples)
python main.py \
    --experiment manager \
    --samples 0

echo "Manager Agent evaluation completed!"
echo "Results saved to results/ablation_studies/manager/"
