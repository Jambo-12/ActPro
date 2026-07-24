import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "qwen3.6-max",
    "qwen_plus",
    "deepseek_v3",
    "deepseek_r1",
    "llama4_scout",
]
DEFAULT_SETTINGS = ["raw_logs_context", "static_kb", "full_kb"]


REFERENCE_ID_PATTERN = re.compile(r"\b([RH])(\d{3})(?:\s*-\s*(?:[RH])?(\d{3}))?\b")


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def build_gt_map(gt_data):
    return {
        item["id"]: {
            "proactive": bool(item["proactive"]),
            "category": item.get("category", "Unknown"),
            "time": item.get("time", "Unknown"),
            "location": item.get("location", "Unknown"),
            "event": item.get("event", ""),
            "reason": item.get("reason", ""),
        }
        for item in gt_data
    }


def determine_case_type(gt_proactive, agent_proactive):
    if gt_proactive and agent_proactive:
        return "TP"
    if gt_proactive and not agent_proactive:
        return "FN"
    if not gt_proactive and not agent_proactive:
        return "TN"
    return "FP"


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def normalize_reference_id(prefix, number):
    return f"{prefix}{int(number):03d}"


def extract_reference_ids(text):
    ids = []
    for match in REFERENCE_ID_PATTERN.finditer(str(text or "")):
        prefix = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        if end < start or end - start > 20:
            end = start
        for number in range(start, end + 1):
            ids.append(normalize_reference_id(prefix, number))
    return list(dict.fromkeys(ids))


def get_entry_id(entry):
    if not isinstance(entry, dict):
        return None
    return entry.get("rule_id") or entry.get("habit_id") or entry.get("id")


def build_kb_index(entries):
    index = {}
    for entry in entries or []:
        entry_id = get_entry_id(entry)
        if entry_id:
            index[str(entry_id)] = entry
    return index


def select_relevant_references(reference_rationale, static_kb, habit_kb):
    ids = extract_reference_ids(reference_rationale)
    static_index = build_kb_index(static_kb)
    habit_index = build_kb_index(habit_kb)
    static_refs = []
    habit_refs = []
    missing_refs = []

    for ref_id in ids:
        if ref_id.startswith("R"):
            entry = static_index.get(ref_id)
            if entry:
                static_refs.append(entry)
            else:
                missing_refs.append(ref_id)
        elif ref_id.startswith("H"):
            entry = habit_index.get(ref_id)
            if entry:
                habit_refs.append(entry)
            else:
                missing_refs.append(ref_id)

    return {
        "reference_ids": ids,
        "static_refs": static_refs,
        "habit_refs": habit_refs,
        "missing_refs": missing_refs,
    }


def format_reference_snippets(refs):
    if not refs:
        return "None explicitly referenced by the benchmark rationale."
    return json.dumps(refs, indent=2, ensure_ascii=False)


def automatic_evaluation(case_type, context):
    if case_type == "FN":
        return {
            "rationale_match": "none",
            "basis_grounding": "missing",
            "missing_key_elements": ["required proactive response was not generated"],
            "reasoning": (
                "Failure: the agent remained silent even though intervention was required. "
                f"Context: {context['time']} at {context['location']}; {context['event']} "
                f"Reference rationale: {context['reference_rationale']}"
            ),
            "correctness": 0,
            "contextual_relevance": 0,
            "clarity": 0,
            "case_type": "FN",
            "average_score": 0.0,
        }
    if case_type == "TN":
        return {
            "rationale_match": "not_applicable",
            "basis_grounding": "not_applicable",
            "missing_key_elements": [],
            "reasoning": (
                "Success: the agent correctly remained silent for a normal event. "
                f"Reference rationale: {context['reference_rationale']}"
            ),
            "correctness": 100,
            "contextual_relevance": 100,
            "clarity": 100,
            "case_type": "TN",
            "average_score": 100.0,
        }
    raise ValueError(f"Automatic scoring is only defined for FN/TN, got {case_type}")


def build_judge_prompt(context, proposed_action, decision_basis, references, case_type, setting):
    reference_ids = ", ".join(references["reference_ids"]) if references["reference_ids"] else "None"
    missing_refs = ", ".join(references["missing_refs"]) if references["missing_refs"] else "None"
    current_decision = "proactive interaction required" if context["gt_proactive"] else "no proactive interaction required"

    if case_type == "TP":
        case_guidance = """
Case note:
- Intervention is required.
- Reward responses that identify the trigger, stay grounded in the provided event and snippets,
  and offer a concrete next step.
- Penalize generic, unsupported, or hallucinated explanations.
"""
    elif case_type == "FP":
        case_guidance = """
Case note:
- No intervention is needed.
- Penalize unnecessary interruptions, especially when they invent a hazard, rule, habit, deadline,
  or user need.
- A fluent warning is still low quality if it is unnecessary.
"""
    else:
        raise ValueError(f"LLM judging is only defined for TP/FP, got {case_type}")

    prompt = f"""
You are an evaluator of proactive smart-home response quality.

Role: Judge whether one generated response is correct, contextually grounded, and clear.
Input: The current event, the reference rationale, the agent output, the agent setting, and the
relevant static-rule / habit snippets.
Constraint: Use only the provided information. Do not infer extra rules, habits, devices, user
states, or deadlines.

Current event:
- Time: {context['time']}
- Location: {context['location']}
- Event: {context['event']}

Ground-truth reference (benchmark target and rationale):
- Required decision: {current_decision}
- Benchmark category: {context['category']}
- Reference rationale: {context['reference_rationale']}

Agent output (the response being evaluated):
- proactive: true
- proposed_action: {proposed_action if proposed_action else "[empty]"}
- decision_basis: {decision_basis if decision_basis else "[empty]"}

Relevant evidence (retrieved IDs and snippets shown to the judge):
- Reference IDs: {reference_ids}
- Missing IDs: {missing_refs}

Relevant static-rule snippets:
{format_reference_snippets(references["static_refs"])}

Relevant habit snippets:
{format_reference_snippets(references["habit_refs"])}

Scoring:
- Correctness: alignment with the reference rationale, the current event, and the relevant snippets.
- Contextual Relevance: use of the specific time, location, observed activity, and contextual cues.
- Clarity: whether the user-facing action is concise, understandable, polite, and actionable.
- Give high scores only when the response is well grounded in the provided evidence.
- Penalize generic, unsupported, or hallucinated claims.

{case_guidance}

Return strict JSON only:
{{
  "rationale_match": "exact | partial | weak | none | not_applicable",
  "basis_grounding": "supported | partially_supported | unsupported | missing | not_applicable",
  "missing_key_elements": ["short labels for omitted trigger details, or []"],
  "reasoning": "Briefly justify the scores using the reference rationale, proposed_action, decision_basis, and relevant snippets.",
  "correctness": <integer 0-100>,
  "contextual_relevance": <integer 0-100>,
  "clarity": <integer 0-100>
}}
"""
    return prompt


def parse_int_score(value):
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def parse_string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in {None, ""}:
        return []
    return [str(value)]


def parse_scores(payload, case_type):
    scores = {
        "rationale_match": str(payload.get("rationale_match", "not_applicable")),
        "basis_grounding": str(payload.get("basis_grounding", "not_applicable")),
        "missing_key_elements": parse_string_list(payload.get("missing_key_elements", [])),
        "reasoning": str(payload.get("reasoning", "")),
        "correctness": parse_int_score(payload.get("correctness", 0)),
        "contextual_relevance": parse_int_score(
            payload.get("contextual_relevance", payload.get("contextual_alignment", 0))
        ),
        "clarity": parse_int_score(payload.get("clarity", 0)),
        "case_type": case_type,
    }
    for key in ("correctness", "contextual_relevance", "clarity"):
        scores[key] = max(0, min(100, scores[key]))
    scores["average_score"] = round(
        (scores["correctness"] + scores["contextual_relevance"] + scores["clarity"]) / 3,
        2,
    )
    return scores


def parse_json_text(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in judge response", text, 0)


def judge_with_openai(client, judge_model, prompt, use_json_mode):
    request_kwargs = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "You are a fair evaluator. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    if use_json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request_kwargs)
    return parse_json_text(response.choices[0].message.content)


def judge_with_gemini(client_config, judge_model, prompt, use_json_mode):
    base_url = client_config["base_url"].rstrip("/")
    model_path = urllib.parse.quote(judge_model, safe="")
    url = f"{base_url}/v1beta/models/{model_path}:generateContent"

    generation_config = {"temperature": 0.0}
    if use_json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "systemInstruction": {
            "parts": [{"text": "You are a fair evaluator. Return JSON only."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": client_config["api_key"],
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code} at {url}: {body}") from exc

    try:
        text = response_payload["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response: {response_payload}") from exc
    return parse_json_text(text)


def judge_with_anthropic(client_config, judge_model, prompt, use_json_mode):
    del use_json_mode

    base_url = client_config["base_url"].rstrip("/")
    url = f"{base_url}/v1/messages"

    payload = {
        "model": judge_model,
        "max_tokens": 2048,
        "system": "You are a fair evaluator. Return JSON only.",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": client_config["api_key"],
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code} at {url}: {body}") from exc

    try:
        content_blocks = response_payload["content"]
        text = "\n".join(
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        )
    except Exception as exc:
        raise RuntimeError(f"Unexpected Anthropic response: {response_payload}") from exc
    return parse_json_text(text)


def judge_with_llm(client, judge_model, prompt, use_json_mode=True, max_retries=3, sleep_seconds=2.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if isinstance(client, dict):
                if client.get("api_format") == "gemini":
                    return judge_with_gemini(client, judge_model, prompt, use_json_mode)
                if client.get("api_format") == "anthropic":
                    return judge_with_anthropic(client, judge_model, prompt, use_json_mode)
            return judge_with_openai(client, judge_model, prompt, use_json_mode)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(last_error)


def is_done(item, retry_errors):
    evaluation = item.get("evaluation")
    if not evaluation:
        return False
    if retry_errors and evaluation.get("case_type") == "ERROR":
        return False
    return True


def evaluate_file(args, client, static_kb, habit_kb, gt_map, model, setting):
    input_path = args.output_root / setting / f"experiment_{model}_{setting}.json"
    output_path = args.output_root / args.score_dir / setting / f"experiment_{model}_{setting}_scored.json"

    input_data = load_json(input_path)
    existing = []
    if args.resume and not args.force and output_path.exists():
        existing = load_json(output_path)

    existing_by_id = {item.get("id"): item for item in existing if item.get("id")}
    final_results = []
    for item in input_data:
        old_item = existing_by_id.get(item.get("id"))
        if old_item and is_done(old_item, args.retry_errors):
            final_results.append(old_item)
        else:
            final_results.append(item.copy())

    pending_indices = [
        idx
        for idx, item in enumerate(final_results)
        if not is_done(item, args.retry_errors)
    ]
    if args.limit is not None:
        pending_indices = pending_indices[: args.limit]

    print(f"{model}/{setting}: {len(final_results)} items, {len(pending_indices)} pending")

    for idx in tqdm(pending_indices, desc=f"{model}/{setting}"):
        item = final_results[idx]
        item_id = item.get("id")
        gt_info = gt_map.get(item_id)
        if not gt_info:
            item["evaluation"] = {
                "rationale_match": "not_applicable",
                "basis_grounding": "not_applicable",
                "missing_key_elements": [],
                "reasoning": f"No ground-truth item found for {item_id}.",
                "correctness": 0,
                "contextual_relevance": 0,
                "clarity": 0,
                "case_type": "ERROR",
                "average_score": 0.0,
            }
            save_json(final_results, output_path)
            continue

        context = {
            "time": gt_info["time"],
            "location": gt_info["location"],
            "event": gt_info["event"],
            "category": gt_info["category"],
            "reference_rationale": gt_info["reason"],
            "gt_proactive": gt_info["proactive"],
        }
        gt_proactive = gt_info["proactive"]
        agent_proactive = coerce_bool(item.get("proactive", False))
        case_type = determine_case_type(gt_proactive, agent_proactive)

        try:
            if case_type in {"FN", "TN"}:
                evaluation = automatic_evaluation(case_type, context)
            else:
                references = select_relevant_references(
                    context["reference_rationale"],
                    static_kb,
                    habit_kb,
                )
                prompt = build_judge_prompt(
                    context=context,
                    proposed_action=item.get("proposed_action", ""),
                    decision_basis=item.get("decision_basis", ""),
                    references=references,
                    case_type=case_type,
                    setting=setting,
                )
                payload = judge_with_llm(
                    client,
                    args.judge_model,
                    prompt,
                    use_json_mode=not args.disable_json_mode,
                )
                evaluation = parse_scores(payload, case_type)
        except Exception as exc:
            evaluation = {
                "rationale_match": "not_applicable",
                "basis_grounding": "not_applicable",
                "missing_key_elements": [],
                "reasoning": f"Judge error: {exc}",
                "correctness": 0,
                "contextual_relevance": 0,
                "clarity": 0,
                "case_type": "ERROR",
                "average_score": 0.0,
            }

        result_item = item.copy()
        result_item["category"] = gt_info["category"]
        result_item["reference_rationale"] = gt_info["reason"]
        result_item["context_info"] = context
        result_item["gt_proactive"] = gt_proactive
        result_item["reference_ids"] = extract_reference_ids(gt_info["reason"])
        result_item["judge_model"] = args.judge_model
        result_item["judge_api_format"] = args.api_format
        result_item["evaluation"] = evaluation
        final_results[idx] = result_item
        save_json(final_results, output_path)

    save_json(final_results, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Score HomeActEval proactive outputs with a strict GPT judge.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--score-dir",
        default="quality_score_strict",
        help="Directory under output-root for scored files.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--gt-file", type=Path, default=ROOT / "knowledgeBase" / "test_Bench.json")
    parser.add_argument("--static-kb", type=Path, default=ROOT / "knowledgeBase" / "staticKB.json")
    parser.add_argument(
        "--habit-kb",
        type=Path,
        default=ROOT / "knowledgeBase" / "habit_GT.json",
        help=(
            "Habit reference file for judging. Defaults to habit_GT.json because "
            "benchmark rationales are checked against the GT habit IDs."
        ),
    )
    parser.add_argument("--judge-model", default=os.getenv("QUALITY_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Judge API base URL. Defaults to the official OpenAI endpoint.",
    )
    parser.add_argument(
        "--api-format",
        choices=["auto", "openai", "gemini", "anthropic"],
        default="auto",
        help=(
            "Judge API protocol. Use openai for chat.completions-compatible endpoints, "
            "gemini for native Gemini generateContent endpoints, and anthropic for "
            "Anthropic Messages API endpoints. In auto mode, URLs containing /google "
            "use gemini and URLs containing /anthropic use anthropic."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--disable-json-mode",
        action="store_true",
        help="Do not request provider-side JSON mode; still parses JSON from the text response.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume existing scored files.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry items marked as ERROR.")
    parser.add_argument("--force", action="store_true", help="Re-score all items, ignoring existing scored files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file item limit for testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.api_format == "auto":
        base_url_text = str(args.base_url or "").rstrip("/").lower()
        if base_url_text.endswith("/google"):
            args.api_format = "gemini"
        elif base_url_text.endswith("/anthropic"):
            args.api_format = "anthropic"
        else:
            args.api_format = "openai"

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    print(f"Judge API format: {args.api_format}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge base URL: {args.base_url or '[OpenAI SDK default]'}")

    if args.api_format in {"gemini", "anthropic"}:
        if not args.base_url:
            raise RuntimeError(f"--base-url is required when --api-format {args.api_format} is used.")
        client = {
            "api_format": args.api_format,
            "api_key": api_key,
            "base_url": str(args.base_url),
        }
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAI-compatible judging. "
                "Install it in the active environment before running quality assessment."
            ) from exc

        client_kwargs = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)

    static_kb = load_json(args.static_kb)
    habit_kb = load_json(args.habit_kb)
    gt_map = build_gt_map(load_json(args.gt_file))

    outputs = []
    for setting in args.settings:
        for model in args.models:
            outputs.append(evaluate_file(args, client, static_kb, habit_kb, gt_map, model, setting))

    print("Scored files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
