# Data And Feature Preparation

This release focuses on the ASTime and PKUMMD setups used by the released
`ActLLaVA-ASTime` and `ActLLaVA-PKUMMD` checkpoints.

## ASTime Layout

Run commands from the `Act-LLaVA/` directory. The expected local layout is:

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

The repository includes ASTime annotation JSON files. Training videos,
test videos, generated feature tensors, sampled videos, and result files are
local artifacts and are ignored by `.gitignore`.

Download ASTime from:

```text
https://huggingface.co/datasets/Jambo1988/ASTime
```

## Training Features

The released main ASTime model was trained with pre-extracted features saved as:

```text
dataset/ASTime/features/<video_key>.pt
```

This release does not provide pre-extracted features. Generate them locally
from the ASTime videos with the scripts below.

Each `<video_key>` must match the corresponding key in
`dataset/ASTime/annotations/train.json`.

Use the same preprocessing if you regenerate features:

- sample ASTime training videos to 2 FPS;
- keep original video resolution during FFmpeg sampling;
- do not FFmpeg-resize or center-crop for the main ASTime setting;
- extract LLaVA-OneVision/SigLIP visual features with full aspect ratio;
- save the final feature tensors as `bfloat16` `.pt` files.

See `../../docs/ASTIME_PREPROCESSING.md` for the full record.

## From Raw ASTime Videos

```bash
SRC_RAW=/path/to/ASTime/raw/train/videos \
bash scripts/ASTime_preprocess_train.sh 0
```

This runs 2 FPS sampling and feature extraction, then writes:

```text
dataset/ASTime/features/<video_key>.pt
```

## From Existing 2 FPS Videos

If you already have a flat folder of 2 FPS training videos:

```bash
SRC_2FPS=/path/to/flat/2fps/train/videos \
bash scripts/ASTime_prepare_features.sh 0
```

The helper script symlinks the 2 FPS folder into the project, runs feature
extraction, and normalizes the output folder name to lowercase `features`.

## Test Videos

For evaluation, put raw ASTime test videos under:

```text
dataset/ASTime/videos/test
```

`demo/test_dataset.py` samples test videos to the model FPS at runtime and
keeps the original resolution for ASTime.

## PKUMMD Layout

Run commands from the `Act-LLaVA/` directory. The expected local layout is:

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

The released PKUMMD model was trained with:

```text
dataset/PKUMMD/annotations/all_v1.json
dataset/PKUMMD/features/<video_key>.pt
```

This release includes the annotation JSON files but does not provide videos or
pre-extracted feature tensors.

Use the same preprocessing if you regenerate features:

- sample PKUMMD videos to 2 FPS;
- center-crop each frame to the largest square region;
- resize the crop to 384x384;
- extract LLaVA-OneVision/SigLIP visual features;
- save the final feature tensors as `bfloat16` `.pt` files.

See `../../docs/PKUMMD_PREPROCESSING.md` for the full record.

## From Raw PKUMMD Videos

```bash
SRC_RAW=/path/to/PKUMMD/raw/videos \
bash scripts/PKUMMD_preprocess_train.sh 0
```

This writes:

```text
dataset/PKUMMD/features/<video_key>.pt
```

## From Existing PKUMMD 384/2FPS Videos

If you already have a flat folder of PKUMMD videos processed as 2 FPS,
center-crop, 384x384:

```bash
SRC_2FPS_384=/path/to/flat/pkummd/384_2fps/videos \
bash scripts/PKUMMD_prepare_features.sh 0
```

## PKUMMD Test Videos

For evaluation, put PKUMMD test videos under:

```text
dataset/PKUMMD/videos/test
```

`demo/test_dataset.py` samples PKUMMD test videos to 2 FPS, center-crops to
square, and resizes to 384x384 at runtime.
