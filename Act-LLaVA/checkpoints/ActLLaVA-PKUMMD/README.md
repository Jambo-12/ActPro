---
library_name: peft
---
## ActLLaVA-PKUMMD

This is the released PKUMMD LoRA adapter for ActPro.

Base model: `lmms-lab/llava-onevision-qwen2-7b-ov`

Training setup:

- dataset: PKUMMD
- training annotation: `dataset/PKUMMD/annotations/all_v1.json`
- visual preprocessing: 2 FPS, center crop to square, resize to 384x384
- visual features: LLaVA-OneVision/SigLIP features saved as bfloat16 tensors
- training script: `scripts/PKUMMD_train.sh`
- reported GPT semantic evaluation: Macro F1 0.8166, Micro F1 0.8169

### Framework versions


- PEFT 0.4.0
