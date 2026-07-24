#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

LAMBDA="${LAMBDA:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ActLLaVA-ASTime}"
TRAIN_DATASETS="${TRAIN_DATASETS:-xdu_clip_stream_train}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/zero2.json}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LLM_PRETRAINED="${LLM_PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-3}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
REPORT_TO="${REPORT_TO:-tensorboard}"

NUM_GPUS="${NUM_GPUS:-3}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
ADDR="${ADDR:-127.0.0.1}"
PORT="${PORT:-12346}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --live_version live1+ \
    --llm_pretrained "$LLM_PRETRAINED" \
    --train_datasets "$TRAIN_DATASETS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --stream_loss_weight "$LAMBDA" \
    --gradient_checkpointing True \
    --prediction_loss_only False \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps "$LOGGING_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --bf16 True \
    --tf32 True \
    --report_to "$REPORT_TO" \
    --output_dir "$OUTPUT_DIR" \
    --attn_implementation "$ATTN_IMPL"
