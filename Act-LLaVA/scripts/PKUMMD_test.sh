#!/bin/bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-dataset/PKUMMD/videos/test}"
SAM_PATH="${SAM_PATH:-dataset/PKUMMD/sam_test}"
RESULT_DIR="${RESULT_DIR:-dataset/PKUMMD/results_ActLLaVA_PKUMMD}"
CKPT="${CKPT:-checkpoints/ActLLaVA-PKUMMD}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LLM_PRETRAINED="${LLM_PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
GT_PATH="${GT_PATH:-dataset/PKUMMD/annotations/test.json}"
GPT_EVAL_MODEL="${GPT_EVAL_MODEL:-${OPENAI_MODEL:-gpt-4o}}"
GPT_EVAL_CACHE="${GPT_EVAL_CACHE:-${RESULT_DIR%/}_pkummd_gptcache_event_v1.json}"
GPT_EVAL_REPORT="${GPT_EVAL_REPORT:-${RESULT_DIR%/}_pkummd_gptreport_event_v1.json}"

echo "Starting PKUMMD inference..."
python -m demo.test_dataset \
    --data_path "$DATA_PATH" \
    --sam_path "$SAM_PATH" \
    --save_path "$RESULT_DIR" \
    --dataset_type PKUMMD \
    --resume_from_checkpoint "$CKPT" \
    --llm_pretrained "$LLM_PRETRAINED" \
    --attn_implementation "$ATTN_IMPL"

echo "Inference completed successfully."

echo "Starting PKUMMD GPT semantic evaluation..."
python eval/evaluation_PKUMMD_gpt.py \
    --pre_folder "$RESULT_DIR" \
    --gt_path "$GT_PATH" \
    --model "$GPT_EVAL_MODEL" \
    --cache_path "$GPT_EVAL_CACHE" \
    --report_path "$GPT_EVAL_REPORT"

echo "All tasks finished."
