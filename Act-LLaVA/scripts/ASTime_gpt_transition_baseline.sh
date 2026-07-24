#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-${GPT_TRANSITION_MODEL:-${OPENAI_MODEL:-gpt-4o}}}"
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-conversation}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/ASTime/results_gpt_transition_baseline}"
METADATA_DIR="${METADATA_DIR:-dataset/ASTime/results_gpt_transition_baseline_meta}"
CACHE_PATH="${CACHE_PATH:-dataset/ASTime/gpt_transition_cache.jsonl}"
GLOB="${GLOB:-*.txt}"
MIN_STABLE_SECONDS="${MIN_STABLE_SECONDS:-2.0}"
FRAME_FPS="${FRAME_FPS:-1.0}"
VIDEO_PATH_TEMPLATE="${VIDEO_PATH_TEMPLATE:-dataset/ASTime/videos/test/{video_id}.MP4}"
LIMIT_FILES="${LIMIT_FILES:-}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PRINT_PROMPT="${PRINT_PROMPT:-0}"

args=(
    --model "$MODEL"
    --base_url "$BASE_URL"
    --output_schema "$OUTPUT_SCHEMA"
    --output_dir "$OUTPUT_DIR"
    --metadata_dir "$METADATA_DIR"
    --cache_path "$CACHE_PATH"
    --min_stable_seconds "$MIN_STABLE_SECONDS"
    --frame_fps "$FRAME_FPS"
    --video_path_template "$VIDEO_PATH_TEMPLATE"
)

if [[ -n "${CAPTION_FILE:-}" && -n "${CAPTION_DIR:-}" ]]; then
    echo "Set only one of CAPTION_FILE or CAPTION_DIR." >&2
    exit 2
fi

if [[ -n "${CAPTION_FILE:-}" ]]; then
    args+=(--input_file "$CAPTION_FILE")
else
    CAPTION_DIR="${CAPTION_DIR:-dataset/ASTime/baseline_captions}"
    args+=(--input_dir "$CAPTION_DIR" --glob "$GLOB")
fi

if [[ -n "$LIMIT_FILES" ]]; then
    args+=(--limit_files "$LIMIT_FILES")
fi

if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
fi

if [[ "$PRINT_PROMPT" == "1" ]]; then
    args+=(--print_prompt)
fi

python tools/gpt_detect_activity_transitions.py "${args[@]}"
