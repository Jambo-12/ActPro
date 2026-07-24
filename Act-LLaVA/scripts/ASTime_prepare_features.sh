#!/bin/bash
# Prepare ASTime training features with the same preprocessing used by the
# ActLLaVA-ASTime model: 2 FPS videos, original video resolution, and
# LLaVA-OneVision/SigLIP visual features saved as bfloat16 tensors.
#
# Usage:
#   SRC_2FPS=/path/to/flat/2fps/train/videos bash scripts/ASTime_prepare_features.sh 0
#
# The input folder must be flat, and each video basename must match an annotation key.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SRC_2FPS="${SRC_2FPS:-}"
V2FPS="${V2FPS:-dataset/ASTime/videos_2fps}"
OUT_DIR="${OUT_DIR:-dataset/ASTime/features}"
PRETRAINED="${PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
EXTRACTOR="${EXTRACTOR:-simple}"
READ_CHUNK_SIZE="${READ_CHUNK_SIZE:-0}"
GPU="${1:-${GPU:-0}}"

if [ -z "$SRC_2FPS" ]; then
    echo "ERROR: set SRC_2FPS to a flat directory of ASTime training videos sampled at 2 FPS."
    exit 1
fi

# ---------- Link the prepared 2 FPS videos into the project. ----------
echo "=== Link prepared 2 FPS videos ==="
mkdir -p "$(dirname "$V2FPS")"
if [ "$SRC_2FPS" != "$V2FPS" ]; then
    if [ -L "$V2FPS" ]; then
        unlink "$V2FPS"
    elif [ -e "$V2FPS" ]; then
        echo "ERROR: $V2FPS exists and is not a symlink. Move it manually before running this script."
        exit 1
    fi
    ln -s "$SRC_2FPS" "$V2FPS"
fi
echo "2 FPS video count: $(find "$V2FPS" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' \) | wc -l)  (symlink $V2FPS -> $SRC_2FPS)"

# ---------- SigLIP encoding -> .pt features ----------
echo "=== Extract SigLIP features ==="
if [ "$EXTRACTOR" = "simple" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -m data.preprocess.extract_feature_simple \
        --video_dir "$V2FPS" \
        --output_dir "$OUT_DIR" \
        --pretrained "$PRETRAINED" \
        --gpu 0 \
        --read_chunk_size "$READ_CHUNK_SIZE"
elif [ "$EXTRACTOR" = "submitit" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -m data.preprocess.extract_feature \
        --video_dir "$V2FPS" \
        --pretrained "$PRETRAINED" \
        --num_gpus 1

    # extract_feature writes to dataset/ASTime/Features (capital F), while the
    # Xdu data loader reads dataset/ASTime/features (lowercase f).
    if [ -d dataset/ASTime/Features ] && [ ! -e "$OUT_DIR" ]; then
        mv dataset/ASTime/Features "$OUT_DIR"
    fi
else
    echo "ERROR: EXTRACTOR must be 'simple' or 'submitit'."
    exit 1
fi
echo "=== Done. Feature count: $(find "$OUT_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l) ==="
echo "Location: $OUT_DIR/<key>.pt"
