# ASTime GPT Transition Baseline

This release includes the GPT-based ASTime baseline script:

```text
Act-LLaVA/tools/gpt_detect_activity_transitions.py
```

The script converts dense frame-wise caption files into sparse activity
transition predictions. It is useful for comparing ActPro with a caption-first
VLM baseline: first generate one caption per sampled frame, then ask GPT to
select the semantically meaningful activity transition timestamps.

## API Configuration

Use the official OpenAI API endpoint by default:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o
```

`OPENAI_BASE_URL` is optional because the script already defaults to
`https://api.openai.com/v1`. Do not hard-code private API keys into the source
code. You can also pass `--api_key` and `--base_url` directly, but environment
variables are recommended for open-source use.

For this transition detector only, `GPT_TRANSITION_MODEL` is also supported as
a backward-compatible model variable. The priority order is `--model`,
`OPENAI_MODEL`, `GPT_TRANSITION_MODEL`, then `gpt-4o`:

```bash
export GPT_TRANSITION_MODEL=gpt-4o
```

## Input

The input is a `.txt` caption file. Each timestamped line should contain one
frame caption:

```text
0.0s: A person is standing near a table.
1.0s: A person is standing near a table.
2.0s: A person sits down on a chair.
3.0s: A person is sitting on a chair.
```

The file stem is used as the `video_id`. For example:

```text
lsr2-1.txt -> video_id = lsr2-1
```

If your caption files have an extra suffix, use `--strip_input_suffix`. The
default strips `_result`, so `lsr2-1_result.txt` becomes `lsr2-1.json`.

The parser accepts extra continuation lines after a timestamped line and appends
them to the previous caption. Empty lines are ignored.

## Output

The script asks GPT to return transition timestamps only. For each timestamp, it
reuses the original input caption at that frame and writes a prediction JSON.

The default output schema is `assistant_list`, matching the original VLM
comparison sparse prediction format:

```json
[
    {
        "role": "assistant",
        "content": "(Video Time = 2.0s) Assistant: A person sits down on a chair.",
        "time": 2.0,
        "fps": 12.3,
        "cost": 0.81
    }
]
```

For direct compatibility with Act-LLaVA ASTime GPT semantic evaluation, use
`--output_schema conversation`:

```json
{
    "video_path": "dataset/ASTime/videos/test/lsr2-1.MP4",
    "frame_fps": 1.0,
    "conversation": [
        {
            "role": "assistant",
            "content": "(Video Time = 2.0s) Assistant: A person sits down on a chair.",
            "time": 2.0,
            "fps": 12.3,
            "cost": 0.81
        }
    ]
}
```

For `conversation` output, `video_path` is rendered from
`--video_path_template`, whose default is:

```text
dataset/ASTime/videos/test/{video_id}.MP4
```

## Run A Single File

From `Act-LLaVA`:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY

python tools/gpt_detect_activity_transitions.py \
  --input_file /path/to/caption_txts/lsr2-1.txt \
  --output_file dataset/ASTime/results_gpt_transition_baseline/lsr2-1.json \
  --output_schema conversation \
  --cache_path dataset/ASTime/gpt_transition_cache.jsonl \
  --metadata_dir dataset/ASTime/results_gpt_transition_baseline_meta \
  --model gpt-4o
```

## Run A Folder

From `Act-LLaVA`:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY

python tools/gpt_detect_activity_transitions.py \
  --input_dir /path/to/caption_txts \
  --output_dir dataset/ASTime/results_gpt_transition_baseline \
  --output_schema conversation \
  --cache_path dataset/ASTime/gpt_transition_cache.jsonl \
  --metadata_dir dataset/ASTime/results_gpt_transition_baseline_meta \
  --model gpt-4o
```

The generated prediction files can then be evaluated with the ASTime GPT
semantic evaluator:

```bash
python eval/astime_gpt_semantic_eval.py \
  --pre_folder dataset/ASTime/results_gpt_transition_baseline \
  --lab_folder dataset/ASTime/annotations/test \
  --time_mode astime_window \
  --threshold 4 \
  --model gpt-4o \
  --cache_path dataset/ASTime/gpt_semantic_cache.jsonl \
  --output_json dataset/ASTime/results_gpt_transition_baseline_gpt_eval.json
```

## Wrapper Script

A thin shell wrapper is provided:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
CAPTION_DIR=/path/to/caption_txts \
bash scripts/ASTime_gpt_transition_baseline.sh
```

Useful overrides:

```bash
CAPTION_FILE=/path/to/lsr2-1.txt bash scripts/ASTime_gpt_transition_baseline.sh
OUTPUT_DIR=dataset/ASTime/results_my_baseline bash scripts/ASTime_gpt_transition_baseline.sh
MODEL=gpt-4o bash scripts/ASTime_gpt_transition_baseline.sh
DRY_RUN=1 CAPTION_DIR=/path/to/caption_txts bash scripts/ASTime_gpt_transition_baseline.sh
PRINT_PROMPT=1 CAPTION_FILE=/path/to/lsr2-1.txt bash scripts/ASTime_gpt_transition_baseline.sh
```

The wrapper defaults to:

```text
CAPTION_DIR=dataset/ASTime/baseline_captions
OUTPUT_DIR=dataset/ASTime/results_gpt_transition_baseline
METADATA_DIR=dataset/ASTime/results_gpt_transition_baseline_meta
CACHE_PATH=dataset/ASTime/gpt_transition_cache.jsonl
OUTPUT_SCHEMA=conversation
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

## Debugging

Use `--dry_run` before spending API calls:

```bash
python tools/gpt_detect_activity_transitions.py \
  --input_dir /path/to/caption_txts \
  --output_dir dataset/ASTime/results_gpt_transition_baseline \
  --output_schema conversation \
  --dry_run
```

Use `--print_prompt` to inspect the exact system and user prompt for the first
input file:

```bash
python tools/gpt_detect_activity_transitions.py \
  --input_file /path/to/caption_txts/lsr2-1.txt \
  --print_prompt
```

Use `--cache_path` for repeated experiments. The cache is keyed by prompt
version, model name, video id, minimum stability setting, and all input frame
captions. Generated caches, metadata, and prediction folders are local artifacts
and should not be committed to GitHub.
