# ActPro: Activity-driven Proactive Agent for Smart Home Environments

This repository contains the experimental code for activity-driven proactive agents in smart home environments. The codebase consists of two main parts:

- `Act-LLaVA`: robotic-perspective activity captioning model for continuous streaming videos
- `HomeActEval`: habit induction, context retrieval, proactive decision generation, and evaluation

The overall goal is to study when and how a smart home agent should intervene according to observed activities, temporal context, household rules, and user habits.

![Framework](framework.jpg)

## Repository Structure

```text
ActPro-main/
├── README.md
├── framework.jpg
├── requirements.txt
├── Act-LLaVA/
│   ├── train.py
│   ├── scripts/
│   ├── demo/
│   ├── eval/
│   ├── data/
│   ├── dataset/
│   ├── models/
│   └── README.md
└── HomeActEval/
    ├── HomeStream-6Months/
    ├── knowledgeBase/
    └── proactive_experiment/
```

## Components

### Act-LLaVA

`Act-LLaVA` is the video understanding module of the repository. It supports:

- video preprocessing
- feature extraction
- model training
- inference on long videos
- evaluation on `PKUMMD` and `ASTime`

### HomeActEval

`HomeActEval` is used for proactive agent experiments in smart home settings. It includes:

- long-term habit induction from household activity logs
- retrieval of relevant rules and habits given time and location
- proactive decision generation with LLMs
- quality assessment of generated proactive responses

## Installation

Create a Python environment first:

```bash
conda create -n actpro python=3.10 -y
conda activate actpro
```

Install PyTorch according to your CUDA setup. For example:

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

If needed, install `flash-attn` separately:

```bash
pip install flash-attn --no-build-isolation
```

Make sure `FFmpeg` is available in your system `PATH`.

## Quick Start

### Single-video demo

```bash
cd Act-LLaVA

python -m demo.cli_onevideo \
    --resume_from_checkpoint huggingface/Act-LLaVA/ForPKUMMD \
    --video_path demo/videos/0002-L.mp4
```

### Video preprocessing

```bash
cd Act-LLaVA

python -m data.preprocess.ffmpeg \
    --frame_fps 2 \
    --frame_resolution 384 \
    --crop_center True \
    --num_tasks 8 \
    --num_nodes 1 \
    --video_dir dataset/PKUMMD/videos
```

```bash
python -m data.preprocess.extract_feature \
    --video_dir dataset/PKUMMD/videos_2fps
```

### Training

```bash
cd Act-LLaVA

bash scripts/PKUMMD_train.sh
bash scripts/ASTime_train.sh
```

### Evaluation

```bash
cd Act-LLaVA

bash scripts/PKUMMD_test.sh
bash scripts/ASTime_test.sh
```

## HomeActEval

The `HomeActEval` directory provides four main scripts:

### 1. Habit induction

```bash
cd HomeActEval/proactive_experiment
python habit_induction.py
```

This script extracts stable daily and weekly habits from long-term household activity logs.

### 2. Proactive decision generation

```bash
cd HomeActEval/proactive_experiment
python experiment_deepseek_r1.py
```

This script predicts whether proactive intervention is needed for each event and generates the corresponding response.

### 3. Evaluation

```bash
cd HomeActEval/proactive_experiment
python evalute_proactive.py
```

This script computes TP, FP, TN, FN, together with Precision, Recall, F1, and Accuracy.

### 4. Response quality assessment

```bash
cd HomeActEval/proactive_experiment
python "Quality Assessment.py"
```

This script uses an LLM judge to score generated proactive responses.

## Data

### Action understanding datasets

- `ASTime`: 🤗 <https://huggingface.co/datasets/Jambo1988/ASTime>
- `PKUMMD`: please obtain the raw videos from the official source and organize them according to the directory structure expected by the scripts

### HomeActEval resources

- `HomeActEval/HomeStream-6Months/`: long-term household activity logs
- `HomeActEval/knowledgeBase/staticKB.json`: static rule base
- `HomeActEval/knowledgeBase/habit_GT.json`: habit annotations
- `HomeActEval/knowledgeBase/test_Bench_400.json`: proactive intervention benchmark

## Models and Checkpoints

Model organization and example local paths are described in `Act-LLaVA/README.md`. Please replace the checkpoint paths in the example commands with the paths available in your local environment.


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
