# HomeActEval Results

This directory stores the released HomeActEval proactive-interaction result
files, metrics, response-quality judgments, and latency profiles.

## Settings

```text
w/o KB      = current event + recent raw historical activity memory
Static KB   = current event + recent raw memory + retrieved static rules
w/ KB       = current event + recent raw memory + retrieved static rules + retrieved habit KB
```

## Result Files

Primary model outputs are stored by setting:

```text
raw_logs_context/experiment_<model>_raw_logs_context.json
static_kb/experiment_<model>_static_kb.json
full_kb/experiment_<model>_full_kb.json
```

Each normal result record contains:

```json
{
  "id": "E001",
  "model": "deepseek_r1",
  "setting": "full_kb",
  "input_time": "Tuesday 10:15",
  "input_location": "bathroom",
  "input_event": "User slipped and fell, remaining stationary on the floor.",
  "proactive": true,
  "proposed_action": "Concise user-facing assistance message.",
  "decision_basis": "Brief evidence-based reason for the intervention."
}
```

The released result records intentionally do not include debug-only prompt or
retrieval fields. Those fields are written only when explicitly requested with:

```text
--save-raw-response
--save-retrieved-context
```

## Generated Metrics

Precomputed aggregate metrics are not included in this compact release.
Regenerate them from the released result files with the scripts documented in
`../README.md`. The generated decision metrics and response-quality summaries
are written under:

```text
metrics/
```

Latency summaries:

```text
proactive_reasoning_latency/metrics/proactive_reasoning_latency_summary.csv
proactive_reasoning_latency/metrics/proactive_reasoning_latency_table.tex
```

## Inputs Used By The Released Runs

```text
../knowledgeBase/test_Bench.json
../knowledgeBase/staticKB.json
../knowledgeBase/extracted_stable_habits_gpt4o.json
raw_logs_context/context_recent_days.txt
```
