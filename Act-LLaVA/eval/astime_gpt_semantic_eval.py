#!/usr/bin/env python3
"""
ASTime event-level evaluation with GPT semantic caption matching.

This evaluator separates temporal localization from semantic correctness:
1. A prediction is a candidate for a GT event only if its timestamp is inside
   the selected GT time window.
2. GPT judges whether the candidate prediction describes the same core event.
3. True candidate pairs are resolved with one-to-one maximum matching, so one
   prediction cannot create multiple TPs.

The metric is event-level for TP/FP/FN. TN is reported from background time bins
only for completeness and accuracy; P/R/F1 should be the primary numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any


PROMPT_VERSION = "astime_gpt_semantic_v1"

# Configure API settings with environment variables. Do not hard-code private keys.
OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """You are a strict semantic evaluator for activity-caption correctness.

Task:
Compare one ground-truth activity event caption with one model-predicted caption.
Decide whether the prediction describes the same core action/event as the ground truth.

Important:
- You are judging semantic correctness only. Do not judge timing, confidence, grammar, or style.
- Captions may be concise labels or natural-language sentences.
- Respond with JSON only.

Return format:
{"match": true/false, "reason": "short reason"}

Mark TRUE when:
- The core action/event is the same.
- Synonyms, tense changes, articles, minor wording differences, and paraphrases preserve the same event.
- Extra posture, actor, scene, or object context is allowed if it does not change the core event.
- If the GT is posture plus action, such as "standing, making coffee", the posture is framing context; the prediction must match the core action ("making coffee"). A prediction that only says "standing" is not enough.
- If the prediction explicitly includes the GT action together with compatible extra details, it can match this GT. The outer evaluator will still allow this prediction to match at most one GT.

Mark FALSE when:
- The captions are related but not the same action/event.
- The target object, body part, direction, or intent is different.
- The prediction describes only a static posture while the GT is an action.
- The GT is a posture/state or a posture transition, and the prediction does not match that state/transition. For example, "standing" is not the same as "standing up from a seated position".
- The prediction is too generic to verify the GT event.
- The prediction contains multiple competing or alternative actions and it is unclear which one corresponds to the GT.
- The prediction says nothing happened, no action, or is empty.

Special examples:
- GT "taking off jacket" vs prediction "removes jacket": TRUE.
- GT "puts on jacket" vs prediction "removes jacket": FALSE.
- GT "touches head" vs prediction "touches neck": FALSE.
- GT "making coffee" vs prediction "pouring water": TRUE only if the prediction clearly refers to the same coffee-making event; otherwise FALSE.
"""

USER_PROMPT_TEMPLATE = """Ground-truth event caption:
{gt_text}

Predicted caption:
{pred_text}

Optional metadata for audit only, not for deciding semantic correctness:
- video_id: {video_id}
- gt_index: {gt_index}
- gt_time_window: [{gt_start}, {gt_end}]
- gt_timestamp: {gt_timestamp}
- prediction_time: {pred_time}

Does the predicted caption describe the same core action/event as the ground truth?
Respond with JSON only: {{"match": true/false, "reason": "short reason"}}"""

NULL_PREDICTION_PATTERNS = [
    r"^\s*$",
    r"^\s*(no action|no activity|nothing happens|nothing happened|none|null|n/?a)\s*\.?\s*$",
    r"^\s*(there is no action|there are no actions|no significant action)\s*\.?\s*$",
]


@dataclasses.dataclass
class GTEvent:
    video_id: str
    index: int
    text: str
    start_time: float
    timestamp: float
    end_time: float
    flag: int = 0


@dataclasses.dataclass
class Prediction:
    video_id: str
    index: int
    text: str
    raw_text: str
    time: float


@dataclasses.dataclass
class SemanticDecision:
    match: bool
    reason: str
    source: str
    cache_key: str | None = None
    failed: bool = False


@dataclasses.dataclass
class VideoResult:
    video_id: str
    tp: int
    fp: int
    fn: int
    tn: int
    matched_pairs: list[dict[str, Any]]
    unmatched_gt: list[dict[str, Any]]
    unmatched_predictions: list[dict[str, Any]]
    warnings: list[str]


def strip_prediction_prefix(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\(Video Time\s*=\s*[-+]?\d+(?:\.\d+)?s\)\s*Assistant:\s*", "", text)
    text = re.sub(r"^Assistant:\s*", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = strip_prediction_prefix(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'")
    return text


def is_null_prediction(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in NULL_PREDICTION_PATTERNS)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def video_id_from_path(path: Path) -> str:
    return path.stem


def load_gt_events(label_path: Path) -> list[GTEvent]:
    payload = json.load(open(label_path, "r", encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"{label_path} should contain exactly one top-level video key")
    video_id, annos = next(iter(payload.items()))
    events: list[GTEvent] = []
    for idx, anno in enumerate(annos):
        if int(anno.get("flag", 0)) != 0:
            continue
        events.append(
            GTEvent(
                video_id=video_id,
                index=idx,
                text=str(anno.get("text", "")).strip(),
                start_time=safe_float(anno.get("start_time")),
                timestamp=safe_float(anno.get("timestamp")),
                end_time=safe_float(anno.get("end_time")),
                flag=int(anno.get("flag", 0)),
            )
        )
    return events


def load_predictions(pred_path: Path, fallback_video_id: str) -> tuple[list[Prediction], float]:
    payload = json.load(open(pred_path, "r", encoding="utf-8"))
    conversation = payload.get("conversation", [])
    preds: list[Prediction] = []
    horizon = 0.0
    for item in conversation:
        horizon = max(horizon, safe_float(item.get("time"), 0.0))
        if item.get("role") != "assistant":
            continue
        raw = str(item.get("content", ""))
        text = strip_prediction_prefix(raw)
        if is_null_prediction(text):
            continue
        preds.append(
            Prediction(
                video_id=fallback_video_id,
                index=len(preds),
                text=text,
                raw_text=raw,
                time=safe_float(item.get("time"), 0.0),
            )
        )
    return preds, horizon


def event_window(event: GTEvent, args: argparse.Namespace) -> tuple[float, float]:
    if args.time_mode == "interval":
        start, end = event.start_time, event.end_time
    elif args.time_mode == "timestamp_window":
        start, end = event.timestamp - args.threshold, event.timestamp + args.threshold
    elif args.time_mode == "astime_window":
        start = max(event.start_time, event.timestamp - args.threshold)
        end = min(event.end_time, event.timestamp + args.threshold)
    else:
        raise ValueError(f"unknown time_mode: {args.time_mode}")
    if end < start:
        start, end = end, start
    return start, end


def is_time_candidate(event: GTEvent, pred: Prediction, args: argparse.Namespace) -> bool:
    start, end = event_window(event, args)
    # Closed interval, by design: start <= t <= end.
    return start <= pred.time <= end


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def in_any_interval(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in intervals)


def compute_background_tn(
    events: list[GTEvent],
    predictions: list[Prediction],
    args: argparse.Namespace,
    pred_horizon: float,
) -> int:
    if args.tn_bin_seconds <= 0:
        return 0
    gt_windows = merge_intervals([event_window(event, args) for event in events])
    max_gt_time = max((event.end_time for event in events), default=0.0)
    max_pred_time = max((pred.time for pred in predictions), default=0.0)
    horizon = max(max_gt_time, max_pred_time, pred_horizon) + args.background_tail_seconds
    if horizon <= 0:
        return 0
    pred_bins = {int(math.floor(max(pred.time, 0.0) / args.tn_bin_seconds)) for pred in predictions}
    tn = 0
    num_bins = int(math.floor(horizon / args.tn_bin_seconds)) + 1
    for bin_index in range(num_bins):
        t = bin_index * args.tn_bin_seconds
        if in_any_interval(t, gt_windows):
            continue
        if bin_index in pred_bins:
            continue
        tn += 1
    return tn


class GPTSemanticMatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = args.model
        self.cache_path = Path(args.cache_path) if args.cache_path else None
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.api_calls = 0
        self.failures = 0
        self.semantic_checks = 0
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("cache_key")
                if key:
                    self.cache[key] = row

    def _append_cache(self, row: dict[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def cache_key(self, gt_text: str, pred_text: str) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "gt": normalize_text(gt_text),
            "prediction": normalize_text(pred_text),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def judge(self, event: GTEvent, pred: Prediction, start: float, end: float) -> SemanticDecision:
        self.semantic_checks += 1
        key = self.cache_key(event.text, pred.text)
        cached = self.cache.get(key)
        if cached and "match" in cached:
            self.cache_hits += 1
            if self.args.verbose:
                print(
                    f"  cache hit pair={self.semantic_checks} "
                    f"gt={event.video_id}:{event.index} pred={pred.index} match={bool(cached['match'])}",
                    flush=True,
                )
            return SemanticDecision(
                match=bool(cached["match"]),
                reason=str(cached.get("reason", "")),
                source="cache",
                cache_key=key,
            )

        if self.args.dry_run:
            if self.args.verbose:
                print(
                    f"  dry-run pair={self.semantic_checks} "
                    f"gt={event.video_id}:{event.index} pred={pred.index}",
                    flush=True,
                )
            return SemanticDecision(
                match=False,
                reason="dry-run: semantic match not called",
                source="dry_run",
                cache_key=key,
            )

        prompt = USER_PROMPT_TEMPLATE.format(
            gt_text=event.text,
            pred_text=pred.text,
            video_id=event.video_id,
            gt_index=event.index,
            gt_start=start,
            gt_end=end,
            gt_timestamp=event.timestamp,
            pred_time=pred.time,
        )

        if self.args.verbose:
            print(
                f"  GPT pair={self.semantic_checks} "
                f"gt={event.video_id}:{event.index} pred={pred.index} "
                f"t={pred.time:.2f}",
                flush=True,
            )
        decision = self._call_with_retries(prompt)
        decision.cache_key = key
        if not decision.failed or self.args.cache_failed:
            row = {
                "cache_key": key,
                "prompt_version": PROMPT_VERSION,
                "model": self.model,
                "gt_normalized": normalize_text(event.text),
                "prediction_normalized": normalize_text(pred.text),
                "match": decision.match,
                "reason": decision.reason,
                "source": decision.source,
                "failed": decision.failed,
                "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            self.cache[key] = row
            self._append_cache(row)
        return decision

    def _call_with_retries(self, prompt: str) -> SemanticDecision:
        last_error = ""
        for attempt in range(1, self.args.max_retries + 1):
            try:
                payload = self._post_chat_completion(prompt)
                content = payload["choices"][0]["message"]["content"]
                parsed = parse_json_response(content)
                if not isinstance(parsed.get("match"), bool):
                    raise ValueError(f"response JSON lacks boolean match: {content}")
                return SemanticDecision(
                    match=bool(parsed["match"]),
                    reason=str(parsed.get("reason", ""))[:1000],
                    source="gpt",
                )
            except Exception as exc:  # noqa: BLE001 - keep evaluation resilient.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.args.max_retries:
                    time.sleep(self.args.retry_sleep * attempt)
        self.failures += 1
        return SemanticDecision(
            match=False,
            reason=f"GPT evaluation failed after retries; default false. Last error: {last_error}",
            source="failed",
            failed=True,
        )

    def _post_chat_completion(self, prompt: str) -> dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        if not api_key or api_key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
            raise RuntimeError(
                "OpenAI API key is not set. Export OPENAI_API_KEY before using GPT evaluation."
            )
        base_url = (os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        url = f"{base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.args.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.api_calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {message}") from exc


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def min_cost_max_matching(
    num_gt: int,
    num_pred: int,
    edges: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    """Return max-cardinality, min-cost matching for GT->prediction edges."""
    if not edges:
        return []

    source = 0
    gt_offset = 1
    pred_offset = gt_offset + num_gt
    sink = pred_offset + num_pred
    n_nodes = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for i in range(num_gt):
        add_edge(source, gt_offset + i, 1, 0)
    for j in range(num_pred):
        add_edge(pred_offset + j, sink, 1, 0)
    for i, j, cost in edges:
        add_edge(gt_offset + i, pred_offset + j, 1, cost)

    flow = 0
    while True:
        dist = [math.inf] * n_nodes
        in_queue = [False] * n_nodes
        parent: list[tuple[int, int] | None] = [None] * n_nodes
        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for edge_index, edge in enumerate(graph[u]):
                v, cap, cost, _rev = edge
                if cap <= 0 or dist[u] + cost >= dist[v]:
                    continue
                dist[v] = dist[u] + cost
                parent[v] = (u, edge_index)
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
        if parent[sink] is None:
            break
        v = sink
        while v != source:
            u, edge_index = parent[v]  # type: ignore[misc]
            edge = graph[u][edge_index]
            rev_index = edge[3]
            edge[1] -= 1
            graph[v][rev_index][1] += 1
            v = u
        flow += 1

    matched: list[tuple[int, int]] = []
    for i in range(num_gt):
        u = gt_offset + i
        for edge in graph[u]:
            v, cap, _cost, _rev = edge
            if pred_offset <= v < pred_offset + num_pred and cap == 0:
                matched.append((i, v - pred_offset))
    return matched


def evaluate_video(
    video_id: str,
    events: list[GTEvent],
    predictions: list[Prediction],
    pred_horizon: float,
    matcher: GPTSemanticMatcher,
    args: argparse.Namespace,
    warnings: list[str],
) -> VideoResult:
    semantic_true_edges: list[tuple[int, int, int]] = []
    pair_decisions: dict[tuple[int, int], SemanticDecision] = {}
    windows = [event_window(event, args) for event in events]

    for gi, event in enumerate(events):
        start, end = windows[gi]
        for pi, pred in enumerate(predictions):
            if not (start <= pred.time <= end):
                continue
            decision = matcher.judge(event, pred, start, end)
            pair_decisions[(gi, pi)] = decision
            if decision.match:
                distance_cost = int(round(abs(pred.time - event.timestamp) * 1000))
                semantic_true_edges.append((gi, pi, distance_cost))

    matched_indices = min_cost_max_matching(len(events), len(predictions), semantic_true_edges)
    matched_set = set(matched_indices)
    matched_gt = {gi for gi, _pi in matched_indices}
    matched_pred = {pi for _gi, pi in matched_indices}

    matched_pairs: list[dict[str, Any]] = []
    for gi, pi in sorted(matched_indices, key=lambda item: (events[item[0]].index, predictions[item[1]].index)):
        event = events[gi]
        pred = predictions[pi]
        decision = pair_decisions[(gi, pi)]
        start, end = windows[gi]
        matched_pairs.append(
            {
                "gt_index": event.index,
                "gt_text": event.text,
                "gt_timestamp": event.timestamp,
                "gt_window": [start, end],
                "prediction_index": pred.index,
                "prediction_time": pred.time,
                "prediction_text": pred.text,
                "semantic_reason": decision.reason,
                "semantic_source": decision.source,
                "cache_key": decision.cache_key,
            }
        )

    unmatched_gt = []
    for gi, event in enumerate(events):
        if gi in matched_gt:
            continue
        start, end = windows[gi]
        candidates = []
        for pi, pred in enumerate(predictions):
            if start <= pred.time <= end:
                decision = pair_decisions.get((gi, pi))
                candidates.append(
                    {
                        "prediction_index": pred.index,
                        "prediction_time": pred.time,
                        "prediction_text": pred.text,
                        "semantic_match": decision.match if decision else None,
                        "semantic_reason": decision.reason if decision else "",
                    }
                )
        unmatched_gt.append(
            {
                "gt_index": event.index,
                "gt_text": event.text,
                "gt_timestamp": event.timestamp,
                "gt_window": [start, end],
                "time_candidates": candidates,
            }
        )

    unmatched_predictions = []
    for pi, pred in enumerate(predictions):
        if pi in matched_pred:
            continue
        candidate_gt = []
        for gi, event in enumerate(events):
            start, end = windows[gi]
            if start <= pred.time <= end:
                decision = pair_decisions.get((gi, pi))
                candidate_gt.append(
                    {
                        "gt_index": event.index,
                        "gt_text": event.text,
                        "semantic_match": decision.match if decision else None,
                        "semantic_reason": decision.reason if decision else "",
                    }
                )
        unmatched_predictions.append(
            {
                "prediction_index": pred.index,
                "prediction_time": pred.time,
                "prediction_text": pred.text,
                "candidate_gt": candidate_gt,
            }
        )

    tp = len(matched_set)
    fn = len(events) - tp
    fp = len(predictions) - tp
    tn = compute_background_tn(events, predictions, args, pred_horizon)
    return VideoResult(
        video_id=video_id,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        matched_pairs=matched_pairs,
        unmatched_gt=unmatched_gt,
        unmatched_predictions=unmatched_predictions,
        warnings=warnings,
    )


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def f1_score(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fp + fn + tn
    return (tp + tn) / denom if denom > 0 else 0.0


def metric_dict(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return {
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1-Score": round(f1_score(p, r), 4),
        "Accuracy": round(accuracy(tp, fp, fn, tn), 4),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ASTime predictions with GPT semantic caption matching."
    )
    parser.add_argument("--pre_folder", default=None, help="Folder containing prediction JSON files.")
    parser.add_argument("--lab_folder", default=None, help="Folder containing ASTime label JSON files.")
    parser.add_argument(
        "--time_mode",
        choices=["astime_window", "timestamp_window", "interval"],
        default="astime_window",
        help=(
            "Temporal candidate rule. astime_window uses "
            "[max(start, timestamp-threshold), min(end, timestamp+threshold)]. "
            "timestamp_window uses [timestamp-threshold, timestamp+threshold]. "
            "interval uses [start_time, end_time]."
        ),
    )
    parser.add_argument("--threshold", type=float, default=4.0, help="Seconds for *_window modes.")
    parser.add_argument("--model", default=os.environ.get("GPT_EVAL_MODEL", "gpt-4o"))
    parser.add_argument(
        "--cache_path",
        default="experiments/astime_videollm_online/gpt_semantic_cache.jsonl",
        help="JSONL cache keyed by normalized GT/prediction pair plus prompt/model version.",
    )
    parser.add_argument("--output_json", default=None, help="Optional path to write summary and details JSON.")
    parser.add_argument("--per_video", action="store_true", help="Print per-video metrics.")
    parser.add_argument("--dry_run", action="store_true", help="Parse files and temporal candidates without calling GPT.")
    parser.add_argument("--print_prompt", action="store_true", help="Print the exact system prompt and one prompt template.")
    parser.add_argument("--limit_videos", type=int, default=None, help="Evaluate only the first N label videos.")
    parser.add_argument("--verbose", action="store_true", help="Print every GPT/cache pair decision request.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--cache_failed", action="store_true", help="Cache failed GPT calls as false.")
    parser.add_argument(
        "--tn_bin_seconds",
        type=float,
        default=0.5,
        help="Background bin size for TN/accuracy. Set <=0 to disable TN.",
    )
    parser.add_argument(
        "--background_tail_seconds",
        type=float,
        default=0.0,
        help="Extra background duration after the last GT/prediction for TN bins.",
    )
    parser.add_argument(
        "--missing_label_policy",
        choices=["skip", "fp"],
        default="skip",
        help="What to do with prediction files that do not have a matching label JSON.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_prompt:
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        print("===== USER PROMPT TEMPLATE =====")
        print(USER_PROMPT_TEMPLATE)
        return

    if not args.pre_folder or not args.lab_folder:
        raise SystemExit("--pre_folder and --lab_folder are required unless --print_prompt is used")

    pre_folder = Path(args.pre_folder)
    lab_folder = Path(args.lab_folder)
    if not pre_folder.exists():
        raise FileNotFoundError(f"pre_folder does not exist: {pre_folder}")
    if not lab_folder.exists():
        raise FileNotFoundError(f"lab_folder does not exist: {lab_folder}")

    matcher = GPTSemanticMatcher(args)
    label_files = {video_id_from_path(path): path for path in sorted(lab_folder.glob("*.json"))}
    pred_files = {video_id_from_path(path): path for path in sorted(pre_folder.glob("*.json"))}
    print("starting ASTime GPT semantic evaluation", flush=True)
    print(f"pre_folder = {pre_folder}", flush=True)
    print(f"lab_folder = {lab_folder}", flush=True)
    print(f"time_mode = {args.time_mode} | threshold = {args.threshold}", flush=True)
    print(f"model = {args.model}", flush=True)
    print(f"cache_path = {matcher.cache_path}", flush=True)
    print(f"label files = {len(label_files)} | prediction files = {len(pred_files)}", flush=True)
    if args.dry_run:
        print("dry_run = true; GPT will not be called", flush=True)

    results: list[VideoResult] = []
    global_warnings: list[str] = []

    label_items = list(label_files.items())
    if args.limit_videos is not None:
        label_items = label_items[: args.limit_videos]

    for video_number, (video_id, label_path) in enumerate(label_items, start=1):
        warnings: list[str] = []
        events = load_gt_events(label_path)
        pred_path = pred_files.get(video_id)
        if pred_path is None:
            warnings.append(f"missing prediction file for {video_id}; all GT events count as FN")
            predictions: list[Prediction] = []
            pred_horizon = 0.0
        else:
            predictions, pred_horizon = load_predictions(pred_path, video_id)
        print(
            f"[{video_number}/{len(label_items)}] {video_id}: "
            f"GT={len(events)} predictions={len(predictions)}",
            flush=True,
        )
        before_checks = matcher.semantic_checks
        before_calls = matcher.api_calls
        before_cache = matcher.cache_hits
        result = evaluate_video(video_id, events, predictions, pred_horizon, matcher, args, warnings)
        results.append(result)
        print(
            f"[{video_number}/{len(label_items)}] {video_id} done: "
            f"TP/FP/FN/TN={result.tp}/{result.fp}/{result.fn}/{result.tn} | "
            f"semantic_pairs={matcher.semantic_checks - before_checks} | "
            f"GPT_calls={matcher.api_calls - before_calls} | "
            f"cache_hits={matcher.cache_hits - before_cache}",
            flush=True,
        )

    extra_pred_ids = sorted(set(pred_files) - set(label_files))
    for video_id in extra_pred_ids:
        message = f"prediction file has no matching label and was {args.missing_label_policy}: {video_id}"
        global_warnings.append(message)
        if args.missing_label_policy == "fp":
            predictions, pred_horizon = load_predictions(pred_files[video_id], video_id)
            results.append(
                evaluate_video(
                    video_id,
                    events=[],
                    predictions=predictions,
                    pred_horizon=pred_horizon,
                    matcher=matcher,
                    args=args,
                    warnings=[message],
                )
            )

    totals = {
        "TP": sum(r.tp for r in results),
        "FP": sum(r.fp for r in results),
        "FN": sum(r.fn for r in results),
        "TN": sum(r.tn for r in results),
    }
    micro = metric_dict(totals["TP"], totals["FP"], totals["FN"], totals["TN"])

    per_video_metrics = []
    for r in results:
        metrics = metric_dict(r.tp, r.fp, r.fn, r.tn)
        per_video_metrics.append(metrics)
        if args.per_video:
            print(
                f"{r.video_id:18s} "
                f"P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
                f"F1={metrics['F1-Score']:.4f} "
                f"TP/FP/FN/TN={r.tp}/{r.fp}/{r.fn}/{r.tn}",
                flush=True,
            )

    if per_video_metrics:
        macro = {
            key: round(sum(m[key] for m in per_video_metrics) / len(per_video_metrics), 4)
            for key in ["Precision", "Recall", "F1-Score", "Accuracy"]
        }
    else:
        macro = {"Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "Accuracy": 0.0}

    summary: dict[str, Any] = {
        "protocol": "ASTime GPT semantic event-level evaluation",
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "pre_folder": str(pre_folder),
        "lab_folder": str(lab_folder),
        "time_mode": args.time_mode,
        "threshold": args.threshold,
        "tn_bin_seconds": args.tn_bin_seconds,
        "counts": totals,
        "micro": micro,
        "macro": macro,
        "num_videos": len(results),
        "semantic_pair_checks": matcher.semantic_checks,
        "gpt_api_calls": matcher.api_calls,
        "cache_hits": matcher.cache_hits,
        "gpt_failures": matcher.failures,
        "cache_path": str(matcher.cache_path) if matcher.cache_path else None,
        "warnings": global_warnings + [w for r in results for w in r.warnings],
    }

    print("******** ASTime GPT Semantic Evaluation ********", flush=True)
    print(f"pre_folder = {pre_folder}", flush=True)
    print(f"lab_folder = {lab_folder}", flush=True)
    print(f"time_mode = {args.time_mode} | threshold = {args.threshold}", flush=True)
    print(f"videos = {len(results)}", flush=True)
    print(f"counts TP/FP/FN/TN = {totals['TP']}/{totals['FP']}/{totals['FN']}/{totals['TN']}", flush=True)
    print(f"MICRO = {micro}", flush=True)
    print(f"MACRO = {macro}", flush=True)
    print(
        f"semantic pair checks = {matcher.semantic_checks} | GPT calls = {matcher.api_calls} | "
        f"cache hits = {matcher.cache_hits} | failures = {matcher.failures}",
        flush=True,
    )
    if summary["warnings"]:
        print("Warnings:", flush=True)
        for warning in summary["warnings"]:
            print(f"  - {warning}", flush=True)

    if args.output_json:
        details = []
        for r in results:
            details.append(
                {
                    "video_id": r.video_id,
                    "counts": {"TP": r.tp, "FP": r.fp, "FN": r.fn, "TN": r.tn},
                    "metrics": metric_dict(r.tp, r.fp, r.fn, r.tn),
                    "matched_pairs": r.matched_pairs,
                    "unmatched_gt": r.unmatched_gt,
                    "unmatched_predictions": r.unmatched_predictions,
                    "warnings": r.warnings,
                }
            )
        output = {"summary": summary, "videos": details}
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
