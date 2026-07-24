import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GT_FILE = ROOT_DIR / "knowledgeBase" / "test_Bench.json"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "results"

MODELS = [
    "qwen3.6-max",
    "qwen_plus",
    "deepseek_v3",
    "deepseek_r1",
    "llama4_scout",
]
SETTINGS = ["raw_logs_context", "static_kb", "full_kb"]

MODEL_DISPLAY = {
    "qwen3.6-max": "Qwen3.6-Max",
    "qwen_plus": "Qwen-Plus",
    "deepseek_v3": "DeepSeek-V3",
    "deepseek_r1": "DeepSeek-R1",
    "llama4_scout": "Llama4-scout",
}

SETTING_DISPLAY = {
    "raw_logs_context": "w/o KB",
    "static_kb": "w/ Static KB",
    "full_kb": "w/ KB",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def result_path(output_root, setting, model):
    return output_root / setting / f"experiment_{model}_{setting}.json"


def evaluate_file(gt_data, pred_data):
    gt_map = {item["id"]: item for item in gt_data}
    expected_ids = [item["id"] for item in gt_data]
    expected_set = set(expected_ids)

    stats = Counter({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    cat_stats = defaultdict(lambda: Counter({"TP": 0, "FP": 0, "TN": 0, "FN": 0}))
    processed_ids = set()
    errors = []
    extra_ids = []
    input_mismatches = []

    for pred_item in pred_data:
        pred_id = pred_item.get("id")
        if pred_id not in gt_map:
            extra_ids.append(pred_id)
            continue
        processed_ids.add(pred_id)
        gt_item = gt_map[pred_id]

        if pred_item.get("input_time") != gt_item.get("time"):
            input_mismatches.append(pred_id)
        elif pred_item.get("input_location") != gt_item.get("location"):
            input_mismatches.append(pred_id)
        elif pred_item.get("input_event") != gt_item.get("event"):
            input_mismatches.append(pred_id)

        if "error" in pred_item:
            errors.append(pred_id)
            y_pred = False
        else:
            y_pred = pred_item.get("proactive")
            if not isinstance(y_pred, bool):
                y_pred = False

        y_true = bool(gt_item.get("proactive"))
        category = gt_item.get("category", "Unknown")

        if y_true and y_pred:
            result = "TP"
        elif not y_true and y_pred:
            result = "FP"
        elif y_true and not y_pred:
            result = "FN"
        else:
            result = "TN"

        stats[result] += 1
        cat_stats[category][result] += 1

    missing_ids = [item_id for item_id in expected_ids if item_id not in processed_ids]
    duplicate_ids = [
        item_id
        for item_id, count in Counter(item.get("id") for item in pred_data).items()
        if item_id in expected_set and count > 1
    ]
    order_mismatch = sum(
        1 for pred_id, expected_id in zip([item.get("id") for item in pred_data], expected_ids)
        if pred_id != expected_id
    ) + abs(len(pred_data) - len(expected_ids))

    return {
        "stats": stats,
        "cat_stats": cat_stats,
        "n_records": len(pred_data),
        "n_errors": len(errors),
        "n_missing": len(missing_ids),
        "n_extra": len(extra_ids),
        "n_duplicates": len(duplicate_ids),
        "order_mismatch": order_mismatch,
        "n_input_mismatch": len(input_mismatches),
        "error_ids": errors,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "extra_ids": extra_ids,
        "input_mismatch_ids": input_mismatches,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all(args):
    gt_data = load_json(args.gt_file)
    output_root = Path(args.output_root)

    overall_rows = []
    category_rows = []
    structure_rows = []

    models = args.models or MODELS
    settings = args.settings or SETTINGS

    for setting in settings:
        for model in models:
            path = result_path(output_root, setting, model)
            if not path.exists():
                structure_rows.append(
                    {
                        "setting": setting,
                        "setting_display": SETTING_DISPLAY.get(setting, setting),
                        "model": model,
                        "model_display": MODEL_DISPLAY.get(model, model),
                        "exists": False,
                        "records": 0,
                        "errors": "",
                        "missing": "",
                        "extra": "",
                        "duplicates": "",
                        "order_mismatch": "",
                        "input_mismatch": "",
                        "path": str(path),
                    }
                )
                continue

            pred_data = load_json(path)
            result = evaluate_file(gt_data, pred_data)
            stats = result["stats"]
            metrics = calculate_metrics(stats["TP"], stats["FP"], stats["TN"], stats["FN"])

            overall_rows.append(
                {
                    "setting": setting,
                    "setting_display": SETTING_DISPLAY.get(setting, setting),
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "TP": stats["TP"],
                    "FP": stats["FP"],
                    "TN": stats["TN"],
                    "FN": stats["FN"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "specificity": metrics["specificity"],
                    "records": result["n_records"],
                    "errors": result["n_errors"],
                }
            )

            structure_rows.append(
                {
                    "setting": setting,
                    "setting_display": SETTING_DISPLAY.get(setting, setting),
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "exists": True,
                    "records": result["n_records"],
                    "errors": result["n_errors"],
                    "missing": result["n_missing"],
                    "extra": result["n_extra"],
                    "duplicates": result["n_duplicates"],
                    "order_mismatch": result["order_mismatch"],
                    "input_mismatch": result["n_input_mismatch"],
                    "path": str(path),
                }
            )

            for category, cat_counter in sorted(result["cat_stats"].items()):
                cat_metrics = calculate_metrics(
                    cat_counter["TP"],
                    cat_counter["FP"],
                    cat_counter["TN"],
                    cat_counter["FN"],
                )
                category_rows.append(
                    {
                        "setting": setting,
                        "setting_display": SETTING_DISPLAY.get(setting, setting),
                        "model": model,
                        "model_display": MODEL_DISPLAY.get(model, model),
                        "category": category,
                        "TP": cat_counter["TP"],
                        "FP": cat_counter["FP"],
                        "TN": cat_counter["TN"],
                        "FN": cat_counter["FN"],
                        "precision": cat_metrics["precision"],
                        "recall": cat_metrics["recall"],
                        "f1": cat_metrics["f1"],
                        "accuracy": cat_metrics["accuracy"],
                        "specificity": cat_metrics["specificity"],
                    }
                )

    metrics_dir = output_root / "metrics"
    overall_path = metrics_dir / "overall_metrics.csv"
    category_path = metrics_dir / "category_metrics.csv"
    structure_path = metrics_dir / "structure_audit.csv"

    write_csv(
        overall_path,
        overall_rows,
        [
            "setting",
            "setting_display",
            "model",
            "model_display",
            "TP",
            "FP",
            "TN",
            "FN",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "specificity",
            "records",
            "errors",
        ],
    )
    write_csv(
        category_path,
        category_rows,
        [
            "setting",
            "setting_display",
            "model",
            "model_display",
            "category",
            "TP",
            "FP",
            "TN",
            "FN",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "specificity",
        ],
    )
    write_csv(
        structure_path,
        structure_rows,
        [
            "setting",
            "setting_display",
            "model",
            "model_display",
            "exists",
            "records",
            "errors",
            "missing",
            "extra",
            "duplicates",
            "order_mismatch",
            "input_mismatch",
            "path",
        ],
    )

    print(f"Wrote overall metrics: {overall_path}")
    print(f"Wrote category metrics: {category_path}")
    print(f"Wrote structure audit: {structure_path}")

    print("\nOverall metrics:")
    for row in overall_rows:
        print(
            f"{row['setting']:<16} {row['model']:<14} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} Acc={row['accuracy']:.3f} "
            f"TP={row['TP']} FP={row['FP']} TN={row['TN']} FN={row['FN']}"
        )

    bad_structure = [
        row for row in structure_rows
        if row["exists"] is not True
        or row["records"] != len(gt_data)
        or row["errors"] not in {0, ""}
        or row["missing"] not in {0, ""}
        or row["extra"] not in {0, ""}
        or row["duplicates"] not in {0, ""}
        or row["order_mismatch"] not in {0, ""}
        or row["input_mismatch"] not in {0, ""}
    ]
    if bad_structure:
        print("\nStructure warnings:")
        for row in bad_structure:
            print(row)
    else:
        print("\nStructure check: all files are complete, sorted, and error-free.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HomeActEval proactive results.")
    parser.add_argument("--gt-file", default=str(DEFAULT_GT_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--models", nargs="*", choices=MODELS)
    parser.add_argument("--settings", nargs="*", choices=SETTINGS)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_all(parse_args())
