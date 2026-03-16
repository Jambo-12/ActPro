export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export PYTHONPATH=
DIR=`pwd`

NUM_GPUS=3
NNODES=1
RANK=0
ADDR='127.0.0.1'
PORT=12325

export CUDA_VISIBLE_DEVICES=3,4,5
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    train.py \
    --deepspeed configs/deepspeed/zero2.json \
    --live_version live1+ \
    --train_datasets pku_clip_stream_train \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --stream_loss_weight 0.2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --prediction_loss_only False \
    --save_strategy steps \
    --save_total_limit 10 \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir output/pku_jsonv8_ran123 \
