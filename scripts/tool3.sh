echo "Running Tool3 Evaluation..."

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Run evaluation with full dataset (0 samples = all samples)
python main.py \
    --experiment tool3 \
    --samples 0

echo "Tool3 evaluation completed!"
echo "Results saved to results/ablation_studies/tool3/"