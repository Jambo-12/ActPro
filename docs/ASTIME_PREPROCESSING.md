# ASTime Preprocessing

This document records the preprocessing pipeline used for the released
`ActLLaVA-ASTime` model. Keep these settings unchanged if you want features
that match the released checkpoint.

## Training Features

The released ASTime model was trained from pre-extracted visual features under:

```text
Act-LLaVA/dataset/ASTime/features/<video_key>.pt
```

This release does not provide those feature tensors. Users should download the
ASTime videos and run the preprocessing pipeline below to generate matching
features locally.

Each feature file corresponds to one ASTime training video. The video basename
must match the annotation key in `dataset/ASTime/annotations/train.json`.

The pipeline is:

1. Start from ASTime training videos.
2. Sample videos to 2 FPS.
3. Keep the original video resolution during video sampling. Do not resize or
   center-crop the video file in the ASTime main setting.
4. Use the LLaVA-OneVision / SigLIP vision stack from
   `lmms-lab/llava-onevision-qwen2-7b-ov`.
5. Set `image_aspect_ratio = "full"` in the model config before processing
   frames.
6. Convert processed image tensors to `float16` on GPU.
7. Generate visual tokens with `image_token_generation(..., batch_size=16)`.
8. Save the resulting features as `bfloat16` `.pt` tensors.

The important point is that ASTime videos are not resized to 384x384 by FFmpeg
in the main `ActLLaVA-ASTime` setting. The visual preprocessing and resizing
are handled by the LLaVA-OneVision image processor during feature extraction.

## Prepare 2 FPS Videos

If your ASTime videos are not already sampled to 2 FPS, use:

```bash
cd Act-LLaVA

python -m data.preprocess.downsample_astime_fps \
  --src /path/to/ASTime/raw/train/videos \
  --dst dataset/ASTime/videos_2fps \
  --fps 2
```

`downsample_astime_fps.py` does the following:

- recursively scans the source video directory;
- writes a flat output directory;
- names each output video as `<basename>.mp4`;
- samples to the requested FPS;
- keeps the original resolution by calling FFmpeg with `resolution=None`.

## Extract Features

After you have a flat 2 FPS video directory:

```bash
cd Act-LLaVA

SRC_2FPS=/path/to/flat/2fps/train/videos \
bash scripts/ASTime_prepare_features.sh 0
```

The helper script creates a symlink at `dataset/ASTime/videos_2fps`, runs
`data.preprocess.extract_feature_simple` by default, and writes features to:

```text
dataset/ASTime/features/<video_key>.pt
```

You can override paths and model location:

```bash
SRC_2FPS=/path/to/flat/2fps/train/videos \
OUT_DIR=dataset/ASTime/features \
PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov \
PY=python \
bash scripts/ASTime_prepare_features.sh 0
```

For very long videos, set `READ_CHUNK_SIZE` to process decoded frames in
smaller groups:

```bash
SRC_2FPS=/path/to/flat/2fps/train/videos \
READ_CHUNK_SIZE=64 \
bash scripts/ASTime_prepare_features.sh 0
```

You can also run the single-GPU extractor directly:

```bash
cd Act-LLaVA

CUDA_VISIBLE_DEVICES=0 python -m data.preprocess.extract_feature_simple \
  --video_dir dataset/ASTime/videos_2fps \
  --output_dir dataset/ASTime/features \
  --pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
  --gpu 0
```

`extract_feature_simple.py` uses the same feature extraction logic as
`extract_feature.py`: full aspect ratio, `float16` processed images,
`image_token_generation` with batch size 16, and `bfloat16` saved features.

If you intentionally want the original Submitit-based launcher, run:

```bash
SRC_2FPS=/path/to/flat/2fps/train/videos \
EXTRACTOR=submitit \
bash scripts/ASTime_prepare_features.sh 0
```

## Test-Time Video Processing

During ASTime evaluation, `demo/test_dataset.py` reads raw test videos from:

```text
dataset/ASTime/videos/test
```

For ASTime, it only samples the test videos to the model FPS and keeps the
original resolution:

```text
ffmpeg_resolution = None
```

The temporary sampled videos are written under:

```text
dataset/ASTime/sam_test_2fps
```

They are cache files and do not need to be committed.

## What Not To Use For The Main ASTime Model

The repository also contains crop/resize utilities for later transfer or
ablation experiments. These are not the preprocessing used by
`ActLLaVA-ASTime`:

- `adaptive_crop_astime.py`: samples to 2 FPS, detects the main person,
  crops a full-height square window, then resizes to 384x384.
- `letterbox_resize_tmm.py`: prepares TMM/PKUMMD-style resized videos.

Do not use those scripts when reproducing the released main ASTime result
unless you are intentionally running the crop384 or transfer ablations.
