# PKUMMD Preprocessing

This document records the preprocessing used by the released `ActLLaVA-PKUMMD`
model. Keep these settings unchanged if you want features compatible with the
released checkpoint.

## Training Features

The released PKUMMD model was trained from pre-extracted visual features under:

```text
Act-LLaVA/dataset/PKUMMD/features/<video_key>.pt
```

This GitHub release does not provide pre-extracted PKUMMD features. Generate
them locally from the PKUMMD videos with the scripts below.

Each feature file corresponds to one PKUMMD video. The basename must match the
annotation key in:

```text
Act-LLaVA/dataset/PKUMMD/annotations/all_v1.json
```

For example:

```text
dataset/PKUMMD/features/0291-L.pt
dataset/PKUMMD/features/0291-M.pt
dataset/PKUMMD/features/0291-R.pt
```

## Operations

The released `ActLLaVA-PKUMMD` model uses:

1. Start from PKUMMD videos.
2. Sample videos to 2 FPS.
3. Center-crop each frame to the largest square region.
4. Resize the square crop to 384x384.
5. Decode the processed videos with `torchvision`.
6. Set `model.config.image_aspect_ratio = "full"`.
7. Run the LLaVA-OneVision/SigLIP image processor.
8. Convert processed images to `float16` on GPU.
9. Run `image_token_generation(..., batch_size=16)`.
10. Save the resulting features as `bfloat16` `.pt` tensors.

This is intentionally different from the main ASTime preprocessing. ASTime keeps
the original video resolution during FFmpeg sampling; PKUMMD uses 384x384
center-cropped videos.

## From Raw PKUMMD Videos

Run commands from `Act-LLaVA/`:

```bash
SRC_RAW=/path/to/PKUMMD/raw/videos \
bash scripts/PKUMMD_preprocess_train.sh 0
```

This creates:

```text
dataset/PKUMMD/videos_384_2fps/<video_key>.mp4
dataset/PKUMMD/features/<video_key>.pt
```

The raw input directory may contain `.avi`, `.mp4`, `.MP4`, or `.AVI` files.
Video basenames should be unique because the output directory is flat.

## From Existing 384/2FPS Videos

If you already have a flat directory of PKUMMD videos processed as 2 FPS,
center-crop, 384x384:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
bash scripts/PKUMMD_prepare_features.sh 0
```

The helper script creates a symlink at `dataset/PKUMMD/videos_384_2fps`, runs
`data.preprocess.extract_feature_simple` by default, and writes features to:

```text
dataset/PKUMMD/features/<video_key>.pt
```

You can override paths and model location:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
OUT_DIR=dataset/PKUMMD/features \
PRETRAINED=lmms-lab/llava-onevision-qwen2-7b-ov \
PY=python \
bash scripts/PKUMMD_prepare_features.sh 0
```

For very long videos, set `READ_CHUNK_SIZE`:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
READ_CHUNK_SIZE=64 \
bash scripts/PKUMMD_prepare_features.sh 0
```

If you intentionally want the original Submitit-based feature launcher:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
EXTRACTOR=submitit \
bash scripts/PKUMMD_prepare_features.sh 0
```

## Test-Time Video Processing

During PKUMMD evaluation, `demo/test_dataset.py` reads test videos from:

```text
dataset/PKUMMD/videos/test
```

For `--dataset_type PKUMMD`, the script samples videos to the model FPS,
center-crops to square, and resizes to the model frame resolution. With the
released model defaults, this is 2 FPS and 384x384.

The script also follows the original PKUMMD test split convention and keeps
video IDs 291 through 334.

## GPT Evaluation

The released result was evaluated with:

```bash
python eval/evaluation_PKUMMD_gpt.py \
  --pre_folder dataset/PKUMMD/results_ActLLaVA_PKUMMD \
  --gt_path dataset/PKUMMD/annotations/test.json \
  --model gpt-4o
```

The default metric uses the full ground-truth event interval. You can set
`--window_size` if you intentionally want a timestamp-window variant.
