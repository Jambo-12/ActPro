# Gemini-2.5-Pro Evaluation Control Analysis

This note compares the original GPT-4o strict evaluator
(`quality_score_strict`) with the non-GPT control evaluator
(`quality_score_gemini_2_5_pro`) for HomeActEval.

## Metric Protocol

The aggregation follows `proactive_code/result.py`:

- Response Quality metrics are averaged over the 240 ground-truth proactive
  cases. True positives use the judge score, and false negatives contribute 0.
- Overall Effectiveness is the mean `average_score` over all 400 cases,
  including true negatives.
- Each model/setting has 400 scored records and 0 judge errors.

## Gemini Table

The full LaTeX table is available at:

```text
results/metrics/quality_effectiveness_gemini_2_5_pro_table.tex
```

The generated row-only version is available at:

```text
results/metrics/quality_effectiveness_gemini_2_5_pro_latex.md
```

## Main Findings

Gemini-2.5-Pro reproduces the main conclusion from GPT-4o: adding the full KB
improves every model on all response-quality dimensions and on overall
effectiveness.

Mean score changes from w/o KB to w/ KB:

| Judge | Correctness | Contextual | Clarity | Overall |
|---|---:|---:|---:|---:|
| GPT-4o | +27.46 | +26.59 | +28.41 | +14.79 |
| Gemini-2.5-Pro | +27.88 | +28.89 | +28.88 | +15.44 |

The overall-effectiveness gains are also very close model by model:

| Model | GPT-4o Gain | Gemini Gain | Difference |
|---|---:|---:|---:|
| Qwen3.6-Max | +14.51 | +14.91 | +0.40 |
| Qwen-Plus | +15.63 | +16.82 | +1.19 |
| DeepSeek-V3 | +17.47 | +18.03 | +0.56 |
| DeepSeek-R1 | +17.16 | +17.39 | +0.23 |
| Llama4-Scout | +9.18 | +10.03 | +0.85 |

Across the 10 aggregate model-setting rows, Gemini and GPT-4o are highly
correlated:

| Metric | Pearson | Spearman |
|---|---:|---:|
| Correctness | 0.998 | 0.988 |
| Contextual | 0.998 | 0.976 |
| Clarity | 1.000 | 1.000 |
| Overall | 0.999 | 1.000 |

The model ranking by overall effectiveness is identical under both judges:

- w/o KB: Llama4-Scout > Qwen-Plus > DeepSeek-V3 > Qwen3.6-Max > DeepSeek-R1
- w/ KB: DeepSeek-V3 > Qwen-Plus > DeepSeek-R1 > Llama4-Scout > Qwen3.6-Max

Gemini is slightly more generous on Contextual and Clarity, while it is nearly
identical on Correctness:

| Metric | Mean Gemini - GPT | Mean Absolute Difference | Max Absolute Difference |
|---|---:|---:|---:|
| Correctness | -0.57 | 0.86 | 2.47 |
| Contextual | +2.18 | 2.32 | 4.16 |
| Clarity | +2.90 | 2.90 | 3.57 |
| Overall | +0.94 | 0.96 | 1.59 |

## Suggested Paper Text

To address the reviewer concern about evaluator bias, we additionally evaluated
the response quality using Gemini-2.5-Pro as a non-GPT control judge. The
Gemini-based results are consistent with the GPT-4o evaluation: the full KB
setting improves all five tested models across correctness, contextual
relevance, clarity, and overall effectiveness. The mean overall-effectiveness
gain increases from +14.79 under GPT-4o to +15.44 under Gemini-2.5-Pro, and the
overall rankings are identical between the two judges in both w/o KB and w/ KB
settings. Across all 10 model-setting aggregates, the two judges show very high
agreement (Pearson r = 0.999 and Spearman rho = 1.000 for overall
effectiveness). These results indicate that the observed improvement is not an
artifact of using a GPT-family evaluator.

## Suggested Reviewer Response

We thank the reviewer for the suggestion. We added an evaluation-control study
using Gemini-2.5-Pro as a non-GPT judge. The new results are reported in
Table X. The conclusion remains unchanged: incorporating the full KB improves
all tested models. The average overall-effectiveness gain is +15.44 with
Gemini-2.5-Pro, close to the +14.79 gain measured by GPT-4o. Moreover, the
aggregate scores from the two judges are strongly correlated across all
model-setting pairs (Pearson r = 0.999; Spearman rho = 1.000 for overall
effectiveness), and the model rankings are identical under both judges. This
supports the robustness of our findings and reduces the risk that the reported
trend is specific to a GPT-based evaluator.
