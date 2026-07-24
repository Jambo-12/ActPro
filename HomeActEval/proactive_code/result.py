import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "qwen3.6-max",
    "qwen_plus",
    "deepseek_v3",
    "deepseek_r1",
    "llama4_scout",
]
DEFAULT_SETTINGS = ["raw_logs_context", "full_kb"]

MODEL_DISPLAY = {
    "qwen3.6-max": "Qwen3.6-Max",
    "qwen_plus": "Qwen-Plus",
    "deepseek_v3": "DeepSeek-V3",
    "deepseek_r1": "DeepSeek-R1",
    "llama4_scout": "Llama4-Scout",
}

SETTING_DISPLAY = {
    "raw_logs_context": "w/o KB",
    "static_kb": "w/ Static KB",
    "full_kb": "w/ KB",
}


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return float(np.mean(values)) if values else 0.0


def summarize_file(path):
    data = load_json(path)
    required_scores = {
        "correctness": [],
        "contextual_relevance": [],
        "clarity": [],
    }
    all_scores = []
    case_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}

    for item in data:
        evaluation = item.get("evaluation")
        if not evaluation:
            continue

        case_type = evaluation.get("case_type", "ERROR")
        case_counts[case_type] = case_counts.get(case_type, 0) + 1
        all_scores.append(float(evaluation.get("average_score", 0)))

        # Required response quality is computed over the fixed set of
        # ground-truth proactive-required cases. TP uses the judge score; FN is
        # already scored as 0 by quality_assessment.py. This avoids TP-only
        # selection bias when different settings recall different instances.
        if item.get("gt_proactive") is True:
            required_scores["correctness"].append(float(evaluation.get("correctness", 0)))
            required_scores["contextual_relevance"].append(
                float(evaluation.get("contextual_relevance", evaluation.get("contextual_alignment", 0)))
            )
            required_scores["clarity"].append(float(evaluation.get("clarity", 0)))

    total = sum(case_counts.values())
    return {
        "records": total,
        "required_cases": len(required_scores["correctness"]),
        "tp": case_counts.get("TP", 0),
        "fp": case_counts.get("FP", 0),
        "tn": case_counts.get("TN", 0),
        "fn": case_counts.get("FN", 0),
        "errors": case_counts.get("ERROR", 0),
        "correctness": mean(required_scores["correctness"]),
        "contextual_relevance": mean(required_scores["contextual_relevance"]),
        "clarity": mean(required_scores["clarity"]),
        "overall_effectiveness": mean(all_scores),
    }


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_display",
        "setting",
        "setting_display",
        "records",
        "required_cases",
        "TP",
        "FP",
        "TN",
        "FN",
        "errors",
        "correctness",
        "contextual_relevance",
        "clarity",
        "overall_effectiveness",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_rows(rows):
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["setting"]] = row

    def fmt_pair(raw_value, full_value):
        raw_text = f"{raw_value:.2f}"
        full_text = f"{full_value:.2f}"
        if full_value > raw_value:
            full_text = f"\\textbf{{{full_text}}}"
        elif raw_value > full_value:
            raw_text = f"\\textbf{{{raw_text}}}"
        return raw_text, full_text

    lines = []
    for model in DEFAULT_MODELS:
        model_rows = by_model.get(model, {})
        raw = model_rows.get("raw_logs_context")
        full = model_rows.get("full_kb")
        if not raw or not full:
            continue
        gain = full["overall_effectiveness"] - raw["overall_effectiveness"]
        display = MODEL_DISPLAY.get(model, model)
        raw_correctness, full_correctness = fmt_pair(raw["correctness"], full["correctness"])
        raw_contextual, full_contextual = fmt_pair(
            raw["contextual_relevance"], full["contextual_relevance"]
        )
        raw_clarity, full_clarity = fmt_pair(raw["clarity"], full["clarity"])
        raw_overall, full_overall = fmt_pair(
            raw["overall_effectiveness"], full["overall_effectiveness"]
        )
        lines.append(f"% --- {display} ---")
        lines.append(f"\\multirow{{2}}{{*}}{{{display}}}")
        lines.append(
            " & w/o KB "
            f"& {raw_correctness} & {raw_contextual} "
            f"& {raw_clarity} & {raw_overall} \\\\"
        )
        lines.append(
            " & w/ KB "
            f"& {full_correctness} "
            f"& {full_contextual} "
            f"& {full_clarity} "
            f"& {full_overall}\\inc{{{gain:.2f}}} \\\\"
        )
        lines.append("\\cmidrule(lr){1-6}")
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize proactive quality scores.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--score-dir",
        default="quality_score_strict",
        help="Directory under output-root containing scored JSON files.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <output-root>/metrics/quality_effectiveness.csv.",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="Optional LaTeX row output. Defaults to <output-root>/metrics/quality_effectiveness_latex.md.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for setting in args.settings:
        for model in args.models:
            path = args.output_root / args.score_dir / setting / f"experiment_{model}_{setting}_scored.json"
            if not path.exists():
                print(f"Missing scored file: {path}")
                continue
            summary = summarize_file(path)
            rows.append(
                {
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "setting": setting,
                    "setting_display": SETTING_DISPLAY.get(setting, setting),
                    "records": summary["records"],
                    "required_cases": summary["required_cases"],
                    "TP": summary["tp"],
                    "FP": summary["fp"],
                    "TN": summary["tn"],
                    "FN": summary["fn"],
                    "errors": summary["errors"],
                    "correctness": round(summary["correctness"], 2),
                    "contextual_relevance": round(summary["contextual_relevance"], 2),
                    "clarity": round(summary["clarity"], 2),
                    "overall_effectiveness": round(summary["overall_effectiveness"], 2),
                }
            )

    csv_path = args.csv or args.output_root / "metrics" / "quality_effectiveness.csv"
    latex_path = args.latex or args.output_root / "metrics" / "quality_effectiveness_latex.md"
    write_csv(rows, csv_path)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_text = latex_rows(rows)
    latex_path.write_text(latex_text + "\n", encoding="utf-8")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX rows: {latex_path}")
    print()
    for row in rows:
        print(
            f"{row['model_display']} / {row['setting_display']}: "
            f"C={row['correctness']:.2f}, Contextual Relevance={row['contextual_relevance']:.2f}, "
            f"Clarity={row['clarity']:.2f}, Overall={row['overall_effectiveness']:.2f}, "
            f"errors={row['errors']}"
        )


if __name__ == "__main__":
    main()
