#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Run Inference
echo "Starting Inference Stage..."

python -m demo.test_dataset  \
    --data_path dataset/PKUMMD/videos \
    --sam_path dataset/PKUMMD/videos_2fps  \
    --save_path dataset/PKUMMD/results   \
    --dataset_type PKUMMD \
    --resume_from_checkpoint output/PKUMMD_train

echo "Inference completed successfully."

# 2. Run Evaluation
echo "Starting Evaluation Stage..."
python eval/evaluation_PKUMMD.py

echo "All tasks finished."