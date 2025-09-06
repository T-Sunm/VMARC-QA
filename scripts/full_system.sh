#!/bin/bash

echo "Running Full System..."

# --- Configuration ---
# Number of samples to process. Set to 0 to run on the full dataset.
SAMPLES=3

# Default data paths (relative to the project root).
# These are used if you followed the default data preparation steps.
TEST_JSON_PATH="data/ViVQA-X/ViVQA-X_test.json"
TEST_IMAGE_DIR="data/COCO_Images/val2014/"

# --- Custom Data Paths (Optional) ---
# If your data is located elsewhere, uncomment and modify the lines below.
# For example:
# TEST_JSON_PATH="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json"
# TEST_IMAGE_DIR="/mnt/VLAI_data/COCO_Images/val2014/"

# --- Run Experiment ---
python main.py \
    --experiment full_system \
    --samples "$SAMPLES" \
    --test_json_path "$TEST_JSON_PATH" \
    --test_image_dir "$TEST_IMAGE_DIR"