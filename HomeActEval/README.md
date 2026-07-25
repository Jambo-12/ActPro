# HomeActEval

HomeActEval is the proactive smart-home evaluation component released with
ActPro. It evaluates whether an LLM-based assistant should proactively
intervene for a current smart-home observation, and whether the proposed
intervention is grounded, useful, and concise.

This subdirectory is self-contained for HomeActEval evaluation. It includes the
benchmark cases, static rules, habit knowledge, six-month synthetic activity
logs, final model outputs, aggregate metrics, response-quality judgments, and
the code needed to reproduce the released analyses.

## Directory Layout

```text
HomeActEval/
|-- README.md
|-- knowledgeBase/
|   |-- test_Bench.json
|   |-- staticKB.json
|   |-- extracted_stable_habits_gpt4o.json
|   `-- habit_GT.json
|-- six_months/
|   |-- month_1.json
|   |-- month_2.json
|   |-- month_3.json
|   |-- month_4.json
|   |-- month_5.json
|   `-- month_6.json
|-- proactive_code/
|   |-- run_experiment.py
|   |-- knowledge_retriever.py
|   |-- evalute_proactive.py
|   |-- quality_assessment.py
|   `-- result.py
`-- results/
    |-- raw_logs_context/
    |-- static_kb/
    |-- full_kb/
    |-- quality_score_strict/
    |-- proactive_reasoning_latency/
    `-- metrics/
```

Historical experiment scripts, local caches, plotting helpers, and exploratory
profiling scripts are intentionally not included in the compact release.

## Benchmark Files

`knowledgeBase/test_Bench.json` contains the evaluation cases. Each item has the
following shape:

```json
{
  "id": "E001",
  "category": "Anomaly Detection",
  "time": "Tuesday 10:15",
  "location": "bathroom",
  "event": "User slipped and fell, remaining stationary on the floor.",
  "proactive": true,
  "reason": "Safety hazard: Potential injury detected via pose estimation."
}
```

Additional knowledge files:

```text
knowledgeBase/staticKB.json
```

Static household rules, such as safety constraints, location constraints,
deadlines, and forbidden actions.

```text
knowledgeBase/extracted_stable_habits_gpt4o.json
```

Habit knowledge extracted from the six-month activity logs and used in the
`full_kb` setting.

```text
knowledgeBase/habit_GT.json
```

Reference habit IDs used by the response-quality evaluator when checking
habit-grounded rationales.

```text
six_months/month_*.json
```

The six-month synthetic smart-home activity logs used to build the released
recent raw-log context and habit knowledge.

The released raw-log context used by the model runs is:

```text
results/raw_logs_context/context_recent_days.txt
```

## Evaluation Settings

HomeActEval compares three input settings:

```text
raw_logs_context = current event + recent raw historical activity memory
static_kb        = current event + recent raw memory + retrieved static rules
full_kb          = current event + recent raw memory + retrieved static rules + retrieved habit KB
```

In the paper-style display tables these correspond to:

```text
w/o KB
w/ Static KB
w/ KB
```

The released experiments evaluate five models:

```text
qwen3.6-max
qwen_plus
deepseek_v3
deepseek_r1
llama4_scout
```

The provider configuration is defined in
`proactive_code/run_experiment.py`. API keys are always read from environment
variables or from the optional `--api-key` argument; no private keys are
hard-coded.

## API Keys

Set only the provider keys you need for the model you are running.

```bash
export DASHSCOPE_API_KEY=...
export QWEN_API_KEY=...
export DEEPSEEK_API_KEY=...
export ZHIZENGZENG_API_KEY=...
export OPENAI_COMPAT_API_KEY=...
```

The GPT response-quality evaluator defaults to the official OpenAI endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export QUALITY_JUDGE_MODEL=gpt-4o
```

`OPENAI_BASE_URL` is optional for the default OpenAI setup. Use `--base-url`
only when routing to a compatible endpoint intentionally.

## Run Experiments

Run commands from this directory:

```bash
cd HomeActEval
```

Dry-run one model and one setting without API calls:

```bash
python proactive_code/run_experiment.py \
  --model deepseek_r1 \
  --setting full_kb \
  --dry-run
```

Run one model and one setting:

```bash
python proactive_code/run_experiment.py \
  --model deepseek_r1 \
  --setting full_kb \
  --resume \
  --retry-errors
```

Run all released model/setting combinations:

```bash
python proactive_code/run_experiment.py \
  --model all \
  --setting all \
  --resume \
  --retry-errors
```

Useful options:

```text
--limit N                 process only N new benchmark items
--resume                  continue from an existing output file
--retry-errors            reprocess records that previously failed
--save-raw-response       store raw model responses for debugging
--save-retrieved-context  store retrieved KB snippets for debugging
```

The released result files intentionally do not include `raw_response` or
`retrieved_context`. Those debug fields are written only if explicitly enabled.

## Output Format

Model outputs are written to:

```text
results/<setting>/experiment_<model>_<setting>.json
```

Each normal record has this shape:

```json
{
  "id": "E001",
  "model": "deepseek_r1",
  "setting": "full_kb",
  "input_time": "Tuesday 10:15",
  "input_location": "bathroom",
  "input_event": "User slipped and fell, remaining stationary on the floor.",
  "proactive": true,
  "proposed_action": "I noticed you slipped and fell in the bathroom. Are you okay?",
  "decision_basis": "Safety hazard: user slipped and fell in the bathroom."
}
```

`decision_basis` is an evaluation-only grounding field. It should be a concise
reason for the proactive decision, not a long chain of thought.

## Decision Metrics

Precomputed aggregate decision metrics are not included in this compact
release. Recompute them from the released result files with:

```bash
python proactive_code/evalute_proactive.py \
  --output-root results
```

This writes generated files under `results/metrics/`:

```text
results/metrics/overall_metrics.csv
results/metrics/category_metrics.csv
results/metrics/structure_audit.csv
```

Metric meaning:

- `overall_metrics.csv`: precision, recall, F1, accuracy, specificity, and
  TP/FP/TN/FN counts for each model and setting.
- `category_metrics.csv`: the same metrics broken down by benchmark category.
- `structure_audit.csv`: file completeness and sanity checks, including missing
  records, duplicate IDs, errors, ordering mismatches, and input mismatches.

## Response-Quality Evaluation

The response-quality evaluator scores proactive outputs with a judge model. The
default setup uses GPT through the official OpenAI API.

```bash
export OPENAI_API_KEY=...

python proactive_code/quality_assessment.py \
  --output-root results \
  --score-dir quality_score_strict \
  --judge-model gpt-4o \
  --resume
```

Summarize the scored files when aggregate quality tables are needed:

```bash
python proactive_code/result.py \
  --output-root results \
  --score-dir quality_score_strict
```

This writes generated CSV summary files under `results/metrics/`:

```text
results/metrics/quality_effectiveness.csv
```

`quality_effectiveness.csv` reports:

- `correctness`
- `contextual_relevance`
- `clarity`
- `overall_effectiveness`

By default, `result.py` summarizes `raw_logs_context` and `full_kb`, matching
the released response-quality comparison.

## Released Results

Primary model outputs:

```text
results/raw_logs_context/experiment_<model>_raw_logs_context.json
results/static_kb/experiment_<model>_static_kb.json
results/full_kb/experiment_<model>_full_kb.json
```

GPT strict response-quality judgments:

```text
results/quality_score_strict/raw_logs_context/
results/quality_score_strict/full_kb/
```

Aggregate decision metrics and response-quality summary tables are generated
artifacts and are not included by default. Regenerate them with the evaluation
commands above.

## Latency Artifacts

The released latency profiles and aggregate latency summaries are under:

```text
results/proactive_reasoning_latency/
results/proactive_reasoning_latency/metrics/proactive_reasoning_latency_summary.csv
```

These files are included for audit and reporting. The compact release includes
the final latency artifacts rather than the exploratory profiling scripts.

## Minimal Validation

Run these checks without external API calls:

```bash
python -m py_compile proactive_code/*.py
python proactive_code/run_experiment.py --model deepseek_r1 --setting full_kb --dry-run
python proactive_code/evalute_proactive.py --output-root results
python proactive_code/result.py --output-root results --score-dir quality_score_strict
```

## Privacy And Keys

Do not hard-code API keys in code or result files. The released result records
are designed to contain only benchmark inputs, model decisions, concise
proposed actions, and evaluation fields. Debug-only raw model responses and
retrieved contexts are excluded unless explicitly requested with debug flags.
