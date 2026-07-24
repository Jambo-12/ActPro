#!/bin/bash
# Full PKUMMD training preprocessing for the released ActLLaVA-PKUMMD setup.
#
# Option A: start from raw PKUMMD videos:
#   SRC_RAW=/path/to/PKUMMD/raw/videos bash scripts/PKUMMD_preprocess_train.sh 0
#
# Option B: start from already prepared flat 384/2FPS videos:
#   SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos bash scripts/PKUMMD_preprocess_train.sh 0
#
# Output:
#   dataset/PKUMMD/videos_384_2fps/<video_key>.mp4
#   dataset/PKUMMD/features/<video_key>.pt
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GPU="${1:-${GPU:-0}}"
FPS="${FPS:-2}"
RESOLUTION="${RESOLUTION:-384}"
SRC_RAW="${SRC_RAW:-}"
SRC_2FPS_384="${SRC_2FPS_384:-}"
V2FPS_384="${V2FPS_384:-dataset/PKUMMD/videos_384_2fps}"
OUT_DIR="${OUT_DIR:-dataset/PKUMMD/features}"
PRETRAINED="${PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"

if [ -z "$SRC_2FPS_384" ]; then
    if [ -z "$SRC_RAW" ]; then
        echo "ERROR: set either SRC_RAW or SRC_2FPS_384."
        exit 1
    fi
    echo "=== Step 1/2: preprocess PKUMMD videos to ${FPS} FPS, center-crop, ${RESOLUTION}x${RESOLUTION} ==="
    "$PY" -m data.preprocess.preprocess_pkummd_384_2fps \
        --src "$SRC_RAW" \
        --dst "$V2FPS_384" \
        --fps "$FPS" \
        --resolution "$RESOLUTION"
    SRC_2FPS_384="$V2FPS_384"
else
    echo "=== Step 1/2: using existing flat PKUMMD 384/2FPS videos ==="
fi

echo "=== Step 2/2: extract LLaVA-OneVision/SigLIP features ==="
SRC_2FPS_384="$SRC_2FPS_384" \
V2FPS_384="$V2FPS_384" \
OUT_DIR="$OUT_DIR" \
PRETRAINED="$PRETRAINED" \
PY="$PY" \
bash scripts/PKUMMD_prepare_features.sh "$GPU"
