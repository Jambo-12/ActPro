#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Run Inference
echo "Starting Inference Stage..."
python -m demo.test_dataset \
    --data_path dataset/ASTime/videos/test \
    --sam_path dataset/ASTime/videos_2fps/test \
    --save_path dataset/ASTime/results \
    --dataset_type ASTime   \
    --resume_from_checkpoint output/ASTime_train

echo "Inference completed successfully."

# 2. Run Evaluation
echo "Starting Evaluation Stage..."
python eval/evaluation_ASTime.py

echo "All tasks finished."