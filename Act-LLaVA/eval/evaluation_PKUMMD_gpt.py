"""
PKUMMD event-level evaluation with GPT semantic matching.

Matching rule:
  1. For each GT event, only predictions inside its evaluation interval are candidates.
     By default the interval is start_time ~ end_time. If --window_size is set, the
     interval is clipped to timestamp +/- window_size.
  2. A candidate prediction becomes TP only when GPT judges that the prediction and
     the GT action label describe the same core action/event.
  3. Each prediction can be matched at most once.
  4. Unmatched GT events are FN.
  5. Unmatched predictions are FP. For class-wise precision, GPT maps each unmatched
     prediction to one PKUMMD action class when possible; otherwise it is counted as
     unknown_fp and included in micro precision only.

The script reports standard macro metrics:
  Macro-P  = mean(per-class precision)
  Macro-R  = mean(per-class recall)
  Macro-F1 = mean(per-class F1)

Example:
  python -u eval/evaluation_PKUMMD_gpt.py \
    --pre_folder dataset/PKUMMD/results_pku_jsonv8_all_v2_processed384_2fps \
    --gt_path dataset/PKUMMD/annotations/test.json \
    --api_key "$OPENAI_API_KEY" \
    --base_url "https://api.openai.com/v1" \
    --model "gpt-4o"
"""

import argparse
import hashlib
import json
import os
import time


PROMPT_VERSION = "pkummd_gpt_event_v1"

# Fill these by command line or environment variables. Do not hard-code private keys.
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

UNKNOWN_FP_CLASS = "__unknown_or_other__"

PAIR_SYSTEM_PROMPT = """
You are a strict evaluator for temporal action recognition/captioning.
You are given one ground-truth PKUMMD action label and one model prediction.
Decide whether the model prediction describes the same core physical action/event.

Return true when:
1. The core action/event is the same.
2. Differences are harmless wording changes, tense changes, articles, or synonyms.
3. The prediction contains extra scene/posture details, but the GT action is clearly present.
4. The prediction contains multiple actions and the GT action is clearly one of them.

Return false when:
1. The action identity, object, target body part, direction, or intent differs.
2. The prediction is only a generic posture/state while the GT is a specific action.
3. The prediction is merely related but too vague to confirm the GT action.
4. The prediction is uncertainty, non-action text, or irrelevant.

Respond with strict JSON only: {"match": true} or {"match": false}.
""".strip()

CLASS_SYSTEM_PROMPT = """
You map a model action prediction to one PKUMMD action class.
Choose a listed class only if the prediction clearly describes that action.
If no listed class is clearly described, return null.

Respond with strict JSON only: {"class": "<exact class label from list>"} or {"class": null}.
""".strip()


def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    return text.strip().lower().rstrip(".")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_prediction_files(pre_folder):
    return sorted(name for name in os.listdir(pre_folder) if name.endswith(".json"))


def parse_predictions(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    preds = []
    for item in content.get("conversation", []):
        if item.get("role") != "assistant":
            continue
        caption = item.get("content", "")
        norm = normalize_text(caption)
        if not norm:
            continue
        preds.append(
            {
                "index": len(preds),
                "time": safe_float(item.get("time")),
                "content": caption,
                "norm": norm,
            }
        )
    return preds


def get_eval_interval(gt_item, window_size):
    start = safe_float(gt_item.get("start_time"))
    end = safe_float(gt_item.get("end_time"))
    if end < start:
        start, end = end, start
    if window_size is None:
        return start, end
    timestamp = safe_float(gt_item.get("timestamp"), (start + end) / 2.0)
    return max(start, timestamp - window_size), min(end, timestamp + window_size)


def make_openai_client(api_key, base_url, timeout):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai package is required for GPT evaluation. "
            "Install it in this environment or run with --dry_run."
        ) from exc
    if not api_key:
        raise SystemExit("Missing API key. Provide --api_key or set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class GPTSemanticMatcher:
    def __init__(
        self,
        client,
        model,
        cache_path,
        class_labels,
        max_retries=3,
        retry_sleep=2.0,
        max_tokens=200,
    ):
        self.client = client
        self.model = model
        self.cache_path = cache_path
        self.class_labels = sorted(class_labels)
        self.class_label_set = set(self.class_labels)
        self.normalized_to_class = {normalize_text(label): label for label in self.class_labels}
        self.labels_hash = hashlib.sha256("\n".join(self.class_labels).encode("utf-8")).hexdigest()
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.max_tokens = max_tokens
        self.api_calls = {"pair": 0, "class": 0}

        self.cache = {
            "__prompt_version__": PROMPT_VERSION,
            "__labels_hash__": self.labels_hash,
            "pair": {},
            "class": {},
        }
        self._load_cache()

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded.get("__prompt_version__") != PROMPT_VERSION:
            print(
                f"[info] ignoring old cache prompt version: "
                f"{loaded.get('__prompt_version__')} != {PROMPT_VERSION}"
            )
            return
        if loaded.get("__labels_hash__") != self.labels_hash:
            print("[info] ignoring old class cache because the label set changed.")
            loaded["class"] = {}
        self.cache["pair"] = loaded.get("pair", {})
        self.cache["class"] = loaded.get("class", {})

    def _save_cache(self):
        if not self.cache_path:
            return
        dirname = os.path.dirname(self.cache_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)

    def _chat_json(self, messages):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_tokens,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait = self.retry_sleep * (2 ** attempt)
                print(f"[warn] GPT call failed ({attempt + 1}/{self.max_retries}): {exc}; sleep {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"GPT call failed after {self.max_retries} retries: {last_error}")

    def match_pair(self, gt_label, pred_caption):
        key = json.dumps(
            {"gt": normalize_text(gt_label), "pred": normalize_text(pred_caption)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in self.cache["pair"]:
            return bool(self.cache["pair"][key])

        user_prompt = (
            f'Ground-truth action label: "{normalize_text(gt_label)}"\n'
            f'Model prediction: "{normalize_text(pred_caption)}"\n\n'
            'Return JSON: {"match": true} or {"match": false}'
        )
        data = self._chat_json(
            [
                {"role": "system", "content": PAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        value = bool(data.get("match", False))
        self.api_calls["pair"] += 1
        self.cache["pair"][key] = value
        self._save_cache()
        return value

    def map_prediction_class(self, pred_caption):
        norm_pred = normalize_text(pred_caption)
        if norm_pred in self.normalized_to_class:
            return self.normalized_to_class[norm_pred]

        key = json.dumps(
            {"labels_hash": self.labels_hash, "pred": norm_pred},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in self.cache["class"]:
            return self.cache["class"][key]

        label_list = "\n".join(f"- {label}" for label in self.class_labels)
        user_prompt = (
            "Known PKUMMD action classes:\n"
            f"{label_list}\n\n"
            f'Model prediction: "{norm_pred}"\n\n'
            'Return JSON: {"class": "<exact class label from list>"} or {"class": null}'
        )
        data = self._chat_json(
            [
                {"role": "system", "content": CLASS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        mapped = data.get("class")
        if mapped is not None:
            mapped = normalize_text(mapped)
            mapped = self.normalized_to_class.get(mapped)

        self.api_calls["class"] += 1
        self.cache["class"][key] = mapped
        self._save_cache()
        return mapped


def evaluate_video(video_id, video_gt, preds, matcher, class_stats, window_size):
    matched_pred_indices = set()
    matched_gt_indices = set()
    details = []

    for g_idx, gt_item in enumerate(video_gt):
        gt_cls = normalize_text(gt_item.get("text", ""))
        eval_start, eval_end = get_eval_interval(gt_item, window_size)
        timestamp = safe_float(gt_item.get("timestamp"), (eval_start + eval_end) / 2.0)

        candidates = []
        for p_idx, pred in enumerate(preds):
            if p_idx in matched_pred_indices:
                continue
            if eval_start <= pred["time"] <= eval_end:
                candidates.append((abs(pred["time"] - timestamp), pred["time"], p_idx, pred))
        candidates.sort()

        matched = None
        for _, _, p_idx, pred in candidates:
            if matcher.match_pair(gt_item.get("text", ""), pred["content"]):
                matched = (p_idx, pred)
                break

        if matched is not None:
            p_idx, pred = matched
            class_stats[gt_cls]["tp"] += 1
            matched_pred_indices.add(p_idx)
            matched_gt_indices.add(g_idx)
            details.append(
                {
                    "type": "TP",
                    "video_id": video_id,
                    "gt_index": g_idx,
                    "pred_index": pred["index"],
                    "gt_label": gt_cls,
                    "pred_caption": pred["norm"],
                    "pred_time": pred["time"],
                    "interval": [eval_start, eval_end],
                }
            )
        else:
            class_stats[gt_cls]["fn"] += 1
            details.append(
                {
                    "type": "FN",
                    "video_id": video_id,
                    "gt_index": g_idx,
                    "gt_label": gt_cls,
                    "interval": [eval_start, eval_end],
                }
            )

    unknown_fp = 0
    for p_idx, pred in enumerate(preds):
        if p_idx in matched_pred_indices:
            continue
        mapped_cls = matcher.map_prediction_class(pred["content"])
        if mapped_cls in class_stats:
            class_stats[mapped_cls]["fp"] += 1
            fp_class = mapped_cls
        else:
            unknown_fp += 1
            fp_class = UNKNOWN_FP_CLASS
        details.append(
            {
                "type": "FP",
                "video_id": video_id,
                "pred_index": pred["index"],
                "pred_caption": pred["norm"],
                "pred_time": pred["time"],
                "fp_class": fp_class,
            }
        )

    return unknown_fp, details


def build_summary(class_stats, unknown_fp):
    rows = []
    total_tp = total_fp = total_fn = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0

    for cls in sorted(class_stats):
        s = class_stats[cls]
        p, r, f1 = metric_from_counts(s["tp"], s["fp"], s["fn"])
        rows.append(
            {
                "class": cls,
                "precision": p,
                "recall": r,
                "f1": f1,
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
            }
        )
        total_tp += s["tp"]
        total_fp += s["fp"]
        total_fn += s["fn"]
        macro_p_sum += p
        macro_r_sum += r
        macro_f1_sum += f1

    total_fp_with_unknown = total_fp + unknown_fp
    num_classes = len(class_stats)
    macro_p = macro_p_sum / num_classes if num_classes else 0.0
    macro_r = macro_r_sum / num_classes if num_classes else 0.0
    macro_f1 = macro_f1_sum / num_classes if num_classes else 0.0
    micro_p, micro_r, micro_f1 = metric_from_counts(total_tp, total_fp_with_unknown, total_fn)

    summary = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp_with_unknown,
        "total_fn": total_fn,
        "unknown_fp": unknown_fp,
        "num_classes": num_classes,
    }
    return rows, summary


def print_report(rows, summary, window_size):
    print(f"## PKUMMD GPT Semantic Evaluation (Window Size: {window_size if window_size is not None else 'Full'})")
    print("| Action Class | Precision | Recall | F1 | TP/FP/FN |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for row in rows:
        print(
            f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['tp']}/{row['fp']}/{row['fn']} |"
        )
    print("\n### Summary")
    print(
        f"* **Macro Average**: P={summary['macro_precision']:.4f}, "
        f"R={summary['macro_recall']:.4f}, F1={summary['macro_f1']:.4f} "
        f"(mean per-class F1)"
    )
    print(
        f"* **Micro Average**: P={summary['micro_precision']:.4f}, "
        f"R={summary['micro_recall']:.4f}, F1={summary['micro_f1']:.4f}"
    )
    print(
        f"* **Counts**: TP={summary['total_tp']}, FP={summary['total_fp']}, "
        f"FN={summary['total_fn']}, unknown_fp={summary['unknown_fp']}"
    )


def dry_run(pre_folder, gt_data, window_size):
    pair_keys = set()
    captions = set()
    used_videos = 0
    for name in list_prediction_files(pre_folder):
        video_id = os.path.splitext(name)[0]
        if video_id not in gt_data:
            continue
        used_videos += 1
        preds = parse_predictions(os.path.join(pre_folder, name))
        captions.update(pred["norm"] for pred in preds)
        for gt_item in gt_data[video_id]:
            eval_start, eval_end = get_eval_interval(gt_item, window_size)
            for pred in preds:
                if eval_start <= pred["time"] <= eval_end:
                    pair_keys.add(
                        json.dumps(
                            {"gt": normalize_text(gt_item.get("text", "")), "pred": pred["norm"]},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
    print(
        "[dry-run] "
        f"videos={used_videos} | "
        f"unique_pair_judgements_upper_bound={len(pair_keys)} | "
        f"unique_prediction_class_maps_upper_bound={len(captions)}"
    )


def main():
    parser = argparse.ArgumentParser(description="PKUMMD GPT semantic event-level evaluation")
    parser.add_argument("--pre_folder", default="dataset/PKUMMD/results", help="prediction JSON directory")
    parser.add_argument("--gt_path", default="dataset/PKUMMD/annotations/test.json", help="PKUMMD GT JSON path")
    parser.add_argument(
        "--window_size",
        type=float,
        default=None,
        help="Use timestamp +/- window_size seconds, clipped by start_time/end_time. Default uses full interval.",
    )
    parser.add_argument("--cache_path", default=None, help="GPT cache JSON path")
    parser.add_argument("--report_path", default=None, help="Optional detailed report JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GPT model name")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API key; default reads OPENAI_API_KEY")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="API base URL; default reads OPENAI_BASE_URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="OpenAI client timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true", help="Parse files and estimate GPT calls without API calls")
    args = parser.parse_args()

    if args.cache_path is None:
        args.cache_path = args.pre_folder.rstrip("/") + "_pkummd_gptcache_event_v1.json"
    if args.report_path is None:
        args.report_path = args.pre_folder.rstrip("/") + "_pkummd_gptreport_event_v1.json"

    gt_data = load_gt(args.gt_path)
    all_classes = sorted({normalize_text(item["text"]) for items in gt_data.values() for item in items})

    if args.dry_run:
        dry_run(args.pre_folder, gt_data, args.window_size)
        return

    client = make_openai_client(args.api_key, args.base_url, args.timeout)
    matcher = GPTSemanticMatcher(
        client=client,
        model=args.model,
        cache_path=args.cache_path,
        class_labels=all_classes,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        max_tokens=args.max_tokens,
    )

    class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in all_classes}
    unknown_fp = 0
    details = []
    used_videos = 0
    skipped_missing_gt = 0

    for name in list_prediction_files(args.pre_folder):
        video_id = os.path.splitext(name)[0]
        if video_id not in gt_data:
            skipped_missing_gt += 1
            print(f"[warn] {name}: missing GT; skipped")
            continue
        used_videos += 1
        pred_path = os.path.join(args.pre_folder, name)
        preds = parse_predictions(pred_path)
        video_unknown_fp, video_details = evaluate_video(
            video_id=video_id,
            video_gt=gt_data[video_id],
            preds=preds,
            matcher=matcher,
            class_stats=class_stats,
            window_size=args.window_size,
        )
        unknown_fp += video_unknown_fp
        details.extend(video_details)
        print(f"[done] {video_id}: preds={len(preds)}, unknown_fp={video_unknown_fp}")

    rows, summary = build_summary(class_stats, unknown_fp)
    summary.update(
        {
            "used_videos": used_videos,
            "skipped_missing_gt": skipped_missing_gt,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "gt_path": args.gt_path,
            "pre_folder": args.pre_folder,
            "window_size": args.window_size,
            "api_calls": matcher.api_calls,
            "cache_path": args.cache_path,
            "report_path": args.report_path,
        }
    )

    print_report(rows, summary, args.window_size)

    report = {"summary": summary, "classes": rows, "details": details}
    dirname = os.path.dirname(args.report_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] saved to {args.report_path}")
    print(f"[cache] saved to {args.cache_path}")


if __name__ == "__main__":
    main()
