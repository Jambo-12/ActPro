# Act-LLaVA

This directory contains the video-understanding component used by ActPro. The
current release includes two Act-LLaVA adapter checkpoints:

```text
checkpoints/ActLLaVA-ASTime
checkpoints/ActLLaVA-PKUMMD
```

## Data Layout

Expected ASTime layout:

```text
dataset/ASTime/
├── annotations/
│   ├── train.json
│   └── test/*.json
├── features/
│   └── <video_key>.pt
└── videos/
    └── test/*.MP4
```

Training reads pre-extracted features from `dataset/ASTime/features`.
Evaluation reads raw test videos from `dataset/ASTime/videos/test` and creates
temporary 2 FPS videos under `dataset/ASTime/sam_test_2fps`.
This release does not provide pre-extracted ASTime features; generate them
locally with the preprocessing scripts.

Expected PKUMMD layout:

```text
dataset/PKUMMD/
├── annotations/
│   ├── all_v1.json
│   └── test.json
├── features/
│   └── <video_key>.pt
└── videos/
    └── test/*.avi
```

PKUMMD training reads pre-extracted features from `dataset/PKUMMD/features`.
Evaluation reads test videos from `dataset/PKUMMD/videos/test` and creates
temporary 384/2FPS videos under `dataset/PKUMMD/sam_test_2fps`.

## ASTime Preprocessing

For the released `ActLLaVA-ASTime` model, ASTime videos were sampled to 2 FPS
without FFmpeg resizing or center cropping. The original video resolution is
kept, and resizing is handled by the LLaVA-OneVision image processor during
feature extraction.

From raw training videos:

```bash
SRC_RAW=/path/to/ASTime/raw/train/videos \
bash scripts/ASTime_preprocess_train.sh 0
```

From already prepared flat 2 FPS videos:

```bash
SRC_2FPS=/path/to/flat/2fps/train/videos \
bash scripts/ASTime_preprocess_train.sh 0
```

See `../docs/ASTIME_PREPROCESSING.md` for the detailed preprocessing record.

## PKUMMD Preprocessing

For the released `ActLLaVA-PKUMMD` model, PKUMMD videos were sampled to 2 FPS,
center-cropped to the largest square region, and resized to 384x384 before
feature extraction.

From raw PKUMMD videos:

```bash
SRC_RAW=/path/to/PKUMMD/raw/videos \
bash scripts/PKUMMD_preprocess_train.sh 0
```

From already prepared flat 384/2FPS videos:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
bash scripts/PKUMMD_preprocess_train.sh 0
```

See `../docs/PKUMMD_PREPROCESSING.md` for the detailed preprocessing record.

## Training

```bash
bash scripts/ASTime_train.sh
```

Default settings:

```text
LAMBDA=0.1
TRAIN_DATASETS=xdu_clip_stream_train
LLM_PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov
OUTPUT_DIR=output/ActLLaVA-ASTime
NUM_GPUS=3
```

PKUMMD:

```bash
bash scripts/PKUMMD_train.sh
```

Default settings:

```text
PKU_TRAIN_JSON=all_v1.json
TRAIN_DATASETS=pku_clip_stream_train
STREAM_LOSS_WEIGHT=0.2
LLM_PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov
OUTPUT_DIR=output/ActLLaVA-PKUMMD
NUM_GPUS=3
```

## Testing

```bash
export OPENAI_API_KEY=...
bash scripts/ASTime_test.sh
```

Default settings:

```text
CKPT=checkpoints/ActLLaVA-ASTime
LLM_PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov
RESULT_DIR=dataset/ASTime/results_ActLLaVA_ASTime
LAB_FOLDER=dataset/ASTime/annotations/test
TIME_MODE=astime_window
THRESHOLD=4
GPT_EVAL_MODEL=gpt-4o
```

To evaluate an existing prediction folder:

```bash
python eval/astime_gpt_semantic_eval.py \
  --pre_folder dataset/ASTime/results_ActLLaVA_ASTime \
  --lab_folder dataset/ASTime/annotations/test \
  --time_mode astime_window \
  --threshold 4 \
  --model gpt-4o \
  --cache_path dataset/ASTime/gpt_semantic_cache.jsonl \
  --output_json dataset/ASTime/results_ActLLaVA_ASTime_gpt_eval.json
```

This release keeps only GPT semantic evaluation for ASTime. Legacy word-match
and exact-match evaluators are intentionally omitted.

PKUMMD:

```bash
export OPENAI_API_KEY=...
bash scripts/PKUMMD_test.sh
```

Default settings:

```text
CKPT=checkpoints/ActLLaVA-PKUMMD
LLM_PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov
RESULT_DIR=dataset/PKUMMD/results_ActLLaVA_PKUMMD
GT_PATH=dataset/PKUMMD/annotations/test.json
GPT_EVAL_MODEL=gpt-4o
```

To evaluate an existing PKUMMD prediction folder:

```bash
python eval/evaluation_PKUMMD_gpt.py \
  --pre_folder dataset/PKUMMD/results_ActLLaVA_PKUMMD \
  --gt_path dataset/PKUMMD/annotations/test.json \
  --model gpt-4o
```

## Model

The released checkpoints are PEFT/LoRA adapters trained from:

```text
lmms-lab/llava-onevision-qwen2-7b-ov
```

The adapter files are hosted on Hugging Face:

[Jambo1988/ActLLaVA](https://huggingface.co/Jambo1988/ActLLaVA)

For local inference, download the corresponding adapter and place it under one
of the default checkpoint directories:

```text
checkpoints/ActLLaVA-ASTime
checkpoints/ActLLaVA-PKUMMD
```

You can also keep the adapters elsewhere and pass the path through `CKPT=...`
when running the test scripts.
