echo "Running LSTM-CNN Baseline Evaluation..."

# Set GPU (optional)
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
python experiments/baselines/lstm_cnn.py \
    --json_path "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json" \
    --image_dir "/mnt/VLAI_data/COCO_Images/val2014/" \
    --limit 0 \
    --random_subset \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --gpu 0 \
    --out "results/baselines/lstm_cnn_300samples.json" \
    --seed 42

echo "LSTM-CNN evaluation completed!"
echo "Results saved to results/baselines/lstm_cnn_300samples.json"