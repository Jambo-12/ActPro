---
library_name: peft
---
# ActLLaVA-ASTime

This directory contains the released PEFT/LoRA adapter for the ASTime
Act-LLaVA model.

Base model:

```text
lmms-lab/llava-onevision-qwen2-7b-ov
```

Local checkpoint path used by the release scripts:

```text
Act-LLaVA/checkpoints/ActLLaVA-ASTime
```

Training setting:

```text
stream_loss_weight = 0.1
```

ASTime training features are expected under:

```text
Act-LLaVA/dataset/ASTime/features/<video_key>.pt
```

The released features were produced from 2 FPS ASTime videos with original
video resolution kept during sampling. Spatial processing is handled by the
LLaVA-OneVision/SigLIP image processor during feature extraction.

## Framework Versions

- PEFT 0.4.0
