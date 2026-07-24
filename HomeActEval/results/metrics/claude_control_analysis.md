# Claude-Sonnet-5 Evaluation Control Summary

This note summarizes the HomeActEval proactive response quality results scored
by Claude-Sonnet-5 (`quality_score_claude_sonnet_5`) using the same aggregation
protocol as GPT-4o and Gemini-2.5-Pro.

## Output Files

```text
results/metrics/quality_effectiveness_claude_sonnet_5.csv
results/metrics/quality_effectiveness_claude_sonnet_5_latex.md
results/metrics/quality_effectiveness_claude_sonnet_5_table.tex
```

## Main Result

Claude-Sonnet-5 is stricter than GPT-4o and Gemini-2.5-Pro in absolute score,
but it supports the same main conclusion: the full KB setting improves every
tested model on correctness, contextual relevance, clarity, and overall
effectiveness.

Mean score changes from w/o KB to w/ KB:

| Judge | Correctness | Contextual | Clarity | Overall |
|---|---:|---:|---:|---:|
| GPT-4o | +27.46 | +26.59 | +28.41 | +14.79 |
| Gemini-2.5-Pro | +27.88 | +28.89 | +28.88 | +15.44 |
| Claude-Sonnet-5 | +25.25 | +25.29 | +25.32 | +13.29 |

Mean overall effectiveness by setting:

| Judge | w/o KB | w/ KB | Gain |
|---|---:|---:|---:|
| GPT-4o | 68.62 | 83.41 | +14.79 |
| Gemini-2.5-Pro | 69.23 | 84.67 | +15.44 |
| Claude-Sonnet-5 | 65.56 | 78.85 | +13.29 |

Model ranking by overall effectiveness:

| Setting | GPT-4o | Gemini-2.5-Pro | Claude-Sonnet-5 |
|---|---|---|---|
| w/o KB | Llama4-Scout > Qwen-Plus > DeepSeek-V3 > Qwen3.6-Max > DeepSeek-R1 | Llama4-Scout > Qwen-Plus > DeepSeek-V3 > Qwen3.6-Max > DeepSeek-R1 | Llama4-Scout > Qwen-Plus > DeepSeek-V3 > Qwen3.6-Max > DeepSeek-R1 |
| w/ KB | DeepSeek-V3 > Qwen-Plus > DeepSeek-R1 > Llama4-Scout > Qwen3.6-Max | DeepSeek-V3 > Qwen-Plus > DeepSeek-R1 > Llama4-Scout > Qwen3.6-Max | DeepSeek-V3 > Qwen-Plus > Qwen3.6-Max > DeepSeek-R1 > Llama4-Scout |

## Suggested Text

We further repeated the proactive response quality evaluation with
Claude-Sonnet-5 as an additional non-GPT judge. Claude-Sonnet-5 assigns lower
absolute scores than GPT-4o and Gemini-2.5-Pro, indicating a stricter judging
tendency. Nevertheless, the conclusion remains consistent: using the full KB
improves all five tested models across correctness, contextual relevance,
clarity, and overall effectiveness. The mean overall-effectiveness gain under
Claude-Sonnet-5 is +13.29, close to the gains measured by GPT-4o (+14.79) and
Gemini-2.5-Pro (+15.44). This provides an additional non-GPT evaluation control
for the robustness of the response-quality results.
