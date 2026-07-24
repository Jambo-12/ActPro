#!/bin/bash
# Full ASTime training preprocessing for the released ActLLaVA-ASTime setup.
#
# Option A: start from raw/full-FPS ASTime training videos:
#   SRC_RAW=/path/to/ASTime/raw/train/videos bash scripts/ASTime_preprocess_train.sh 0
#
# Option B: start from an already prepared flat 2 FPS video directory:
#   SRC_2FPS=/path/to/flat/2fps/train/videos bash scripts/ASTime_preprocess_train.sh 0
#
# Output:
#   dataset/ASTime/videos_2fps/<video_key>.mp4
#   dataset/ASTime/features/<video_key>.pt
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GPU="${1:-${GPU:-0}}"
FPS="${FPS:-2}"
SRC_RAW="${SRC_RAW:-}"
SRC_2FPS="${SRC_2FPS:-}"
V2FPS="${V2FPS:-dataset/ASTime/videos_2fps}"
OUT_DIR="${OUT_DIR:-dataset/ASTime/features}"
PRETRAINED="${PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"

if [ -z "$SRC_2FPS" ]; then
    if [ -z "$SRC_RAW" ]; then
        echo "ERROR: set either SRC_RAW or SRC_2FPS."
        exit 1
    fi
    echo "=== Step 1/2: downsample ASTime videos to ${FPS} FPS, keeping original resolution ==="
    "$PY" -m data.preprocess.downsample_astime_fps \
        --src "$SRC_RAW" \
        --dst "$V2FPS" \
        --fps "$FPS"
    SRC_2FPS="$V2FPS"
else
    echo "=== Step 1/2: using existing flat 2 FPS videos ==="
fi

echo "=== Step 2/2: extract LLaVA-OneVision/SigLIP features ==="
SRC_2FPS="$SRC_2FPS" \
V2FPS="$V2FPS" \
OUT_DIR="$OUT_DIR" \
PRETRAINED="$PRETRAINED" \
PY="$PY" \
bash scripts/ASTime_prepare_features.sh "$GPU"
