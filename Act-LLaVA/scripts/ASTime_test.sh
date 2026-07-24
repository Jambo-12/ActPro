#!/bin/bash
set -euo pipefail

LAMBDA="${LAMBDA:-0.1}"
DATA_PATH="${DATA_PATH:-dataset/ASTime/videos/test}"
SAM_PATH="${SAM_PATH:-dataset/ASTime/sam_test}"
RESULT_DIR="${RESULT_DIR:-dataset/ASTime/results_ActLLaVA_ASTime}"
CKPT="${CKPT:-checkpoints/ActLLaVA-ASTime}"
THRESHOLD="${THRESHOLD:-4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LLM_PRETRAINED="${LLM_PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
LAB_FOLDER="${LAB_FOLDER:-dataset/ASTime/annotations/test}"
TIME_MODE="${TIME_MODE:-astime_window}"
GPT_EVAL_MODEL="${GPT_EVAL_MODEL:-${OPENAI_MODEL:-gpt-4o}}"
GPT_EVAL_CACHE="${GPT_EVAL_CACHE:-dataset/ASTime/gpt_semantic_cache.jsonl}"
GPT_EVAL_OUTPUT="${GPT_EVAL_OUTPUT:-${RESULT_DIR%/}_gpt_eval.json}"

# 1. Run Inference
echo "Starting Inference Stage..."
python -m demo.test_dataset \
    --data_path "$DATA_PATH" \
    --sam_path "$SAM_PATH" \
    --save_path "$RESULT_DIR" \
    --dataset_type ASTime \
    --resume_from_checkpoint "$CKPT" \
    --llm_pretrained "$LLM_PRETRAINED" \
    --attn_implementation "$ATTN_IMPL"

echo "Inference completed successfully."

# 2. Run GPT semantic evaluation.
echo "Starting GPT Semantic Evaluation Stage..."
python eval/astime_gpt_semantic_eval.py \
    --pre_folder "$RESULT_DIR" \
    --lab_folder "$LAB_FOLDER" \
    --time_mode "$TIME_MODE" \
    --threshold "$THRESHOLD" \
    --model "$GPT_EVAL_MODEL" \
    --cache_path "$GPT_EVAL_CACHE" \
    --output_json "$GPT_EVAL_OUTPUT"

echo "All tasks finished."
