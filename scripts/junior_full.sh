#!/bin/bash

echo "Running Junior Agent Full Dataset Evaluation..."

# Set GPU
export CUDA_VISIBLE_DEVICES=1


# Run evaluation with full dataset (0 samples = all samples)
python main.py \
    --experiment junior \
    --samples 0

echo "Junior Agent evaluation completed!"
echo "Results saved to results/ablation_studies/junior/"
