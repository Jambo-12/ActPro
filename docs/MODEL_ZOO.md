# Model Zoo

## ActLLaVA-ASTime

The released ASTime model is a PEFT/LoRA adapter trained with
`stream_loss_weight = 0.1`.

Local path in this release workspace:

```text
Act-LLaVA/checkpoints/ActLLaVA-ASTime
```

Files included in the local adapter bundle:

```text
README.md
adapter_config.json
adapter_model.safetensors
added_tokens.json
merges.txt
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
```

The adapter was trained from the base model:

```text
lmms-lab/llava-onevision-qwen2-7b-ov
```

GitHub note: `adapter_model.safetensors` is larger than GitHub's normal file
limit. The public GitHub code release should not commit the raw adapter file.
The released adapter is hosted on Hugging Face:

```text
https://huggingface.co/Jambo1988/ActLLaVA
```

## ASTime Evaluation

Example prediction directory:

```text
Act-LLaVA/dataset/ASTime/results_ActLLaVA_ASTime
```

This release keeps only GPT semantic evaluation for ASTime. Run:

```bash
cd Act-LLaVA
python eval/astime_gpt_semantic_eval.py \
  --pre_folder dataset/ASTime/results_ActLLaVA_ASTime \
  --lab_folder dataset/ASTime/annotations/test \
  --time_mode astime_window \
  --threshold 4 \
  --model gpt-4o
```

## ActLLaVA-PKUMMD

The released PKUMMD model is a PEFT/LoRA adapter trained with
`stream_loss_weight = 0.2`.

Local path in this release workspace:

```text
Act-LLaVA/checkpoints/ActLLaVA-PKUMMD
```

Training setup:

```text
PKU_TRAIN_JSON=all_v1.json
TRAIN_DATASETS=pku_clip_stream_train
preprocessing=2 FPS, center crop, 384x384
```

The adapter was trained from the base model:

```text
lmms-lab/llava-onevision-qwen2-7b-ov
```

GitHub note: `adapter_model.safetensors` is larger than GitHub's normal file
limit. The released adapter is hosted on Hugging Face:

```text
https://huggingface.co/Jambo1988/ActLLaVA
```

## PKUMMD Evaluation

The corresponding prediction directory in the research workspace was:

```text
Act-LLaVA/dataset/PKUMMD/results_PKUMMD_all_v1_processed384_2fps_reextract_20260712
```

GPT semantic evaluation result:

```text
Macro Average: P=0.8081, R=0.8348, F1=0.8166
Micro Average: P=0.8032, R=0.8311, F1=0.8169
Counts: TP=2155, FP=528, FN=438, unknown_fp=2
```

Run:

```bash
cd Act-LLaVA
python eval/evaluation_PKUMMD_gpt.py \
  --pre_folder dataset/PKUMMD/results_ActLLaVA_PKUMMD \
  --gt_path dataset/PKUMMD/annotations/test.json \
  --model gpt-4o
```
