# Act-LLaVA

## 🛠 Installation

**Prerequisites:** Anaconda & Python $\ge$ 3.10

```sh
# 1. Install PyTorch and CUDA
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install Python Dependencies
pip install transformers accelerate deepspeed peft editdistance Levenshtein tensorboard gradio moviepy submitit
pip install flash-attn --no-build-isolation

# 3. Install FFmpeg (Static Release)
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xvf ffmpeg-release-amd64-static.tar.xz && rm ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-*-amd64-static ffmpeg
```

## 🚀 Quick Start (Demo)

Run the CLI demo to test a single video:

```sh
python -m demo.cli_onevideo \
    --resume_from_checkpoint huggingface/Act-LLaVA/ForPKUMMD \
    --video_path demo/videos/0002-L.mp4
```

-----

## 🏗 Training & Evaluation

### 1\. Data Preparation

Follow the instructions in [data/preprocess/](https://www.google.com/search?q=data/preprocess/) to construct your dataset.

### 2\. Execution

| Task | PKUMMD Dataset | ASTime Dataset |
| :--- | :--- | :--- |
| **Training** | `bash scripts/PKUMMD_train.sh` | `bash scripts/ASTime_train.sh` |
| **Evaluation** | `bash scripts/PKUMMD_test.sh` | `bash scripts/ASTime_test.sh` |

-----

## 🗃 Model Zoo

We recommend using **Symbolic Links** to manage model weights within the project root for path consistency.

### 1\. Directory Structure

```
Act-LLaVA/
├── huggingface/
│   ├── Qwen/
│   │   └── Qwen2-7B-Instruct
│   ├── google/
│   │   └── siglip-so400m-patch14-384
│   └── Jambo/
│       ├── ForPKUMMD
│       └── ForASTime
└── ...
```
### 2\. Setup Links

Run the following commands in the project root:

```bash
mkdir -p huggingface
ln -s /your/absolute/path/to/models/* ./huggingface/
```

### 3\. Model Details

| Component | Source (HuggingFace) | Local Path |
| :--- | :--- | :--- |
| **LLM** | `Qwen/Qwen2-7B-Instruct` | `./huggingface/Qwen2-7B-Instruct` |
| **Vision Tower** | `google/siglip-so400m-patch14-384` | `./huggingface/siglip-so400m-patch14-384` |
| **ForPKUMMD** | `Jambo/ForPKUMMD` | `./huggingface/Jambo/ForPKUMMD` |
| **ForASTime** | `Jambo/ForASTime` | `./huggingface/Jambo/ForASTime` |

-----