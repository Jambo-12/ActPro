#!/bin/bash
# Prepare PKUMMD training features for the released ActLLaVA-PKUMMD setup.
#
# Input videos must already be sampled to 2 FPS, center-cropped to a square,
# resized to 384x384, and stored in a flat directory whose basenames match the
# PKUMMD annotation keys.
#
# Usage:
#   SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos bash scripts/PKUMMD_prepare_features.sh 0
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SRC_2FPS_384="${SRC_2FPS_384:-}"
V2FPS_384="${V2FPS_384:-dataset/PKUMMD/videos_384_2fps}"
OUT_DIR="${OUT_DIR:-dataset/PKUMMD/features}"
PRETRAINED="${PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
EXTRACTOR="${EXTRACTOR:-simple}"
READ_CHUNK_SIZE="${READ_CHUNK_SIZE:-0}"
GPU="${1:-${GPU:-0}}"

if [ -z "$SRC_2FPS_384" ]; then
    echo "ERROR: set SRC_2FPS_384 to a flat directory of PKUMMD videos processed as 2FPS center-crop 384."
    exit 1
fi

echo "=== Link prepared PKUMMD 384/2FPS videos ==="
mkdir -p "$(dirname "$V2FPS_384")"
if [ "$SRC_2FPS_384" != "$V2FPS_384" ]; then
    if [ -L "$V2FPS_384" ]; then
        unlink "$V2FPS_384"
    elif [ -e "$V2FPS_384" ]; then
        echo "ERROR: $V2FPS_384 exists and is not a symlink. Move it manually before running this script."
        exit 1
    fi
    ln -s "$SRC_2FPS_384" "$V2FPS_384"
fi
echo "PKUMMD processed video count: $(find "$V2FPS_384" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' \) | wc -l)"

echo "=== Extract LLaVA-OneVision/SigLIP features ==="
if [ "$EXTRACTOR" = "simple" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -m data.preprocess.extract_feature_simple \
        --video_dir "$V2FPS_384" \
        --output_dir "$OUT_DIR" \
        --pretrained "$PRETRAINED" \
        --gpu 0 \
        --read_chunk_size "$READ_CHUNK_SIZE"
elif [ "$EXTRACTOR" = "submitit" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -m data.preprocess.extract_feature \
        --video_dir "$V2FPS_384" \
        --pretrained "$PRETRAINED" \
        --num_gpus 1

    if [ -d dataset/PKUMMD/Features ] && [ ! -e "$OUT_DIR" ]; then
        mv dataset/PKUMMD/Features "$OUT_DIR"
    fi
else
    echo "ERROR: EXTRACTOR must be 'simple' or 'submitit'."
    exit 1
fi

echo "=== Done. Feature count: $(find "$OUT_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l) ==="
echo "Location: $OUT_DIR/<video_key>.pt"
