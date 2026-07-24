import argparse
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_retriever import KnowledgeRetriever


DEFAULT_MODELS = [
    "qwen3.6-max",
    "qwen_plus",
    "deepseek_v3",
    "deepseek_r1",
    "llama4_scout",
]

MODEL_CONFIGS = {
    "qwen3.6-max": {
        "display_name": "qwen3.6-max",
        "model_name": "qwen3.6-max-preview",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "response_format": True,
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False},
    },
    "qwen_plus": {
        "display_name": "Qwen-Plus",
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "response_format": True,
        "temperature": 0.0,
    },
    "deepseek_v3": {
        "display_name": "DeepSeek-V3",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": [
            "DEEPSEEK_API_KEY",
            "ZHIZENGZENG_API_KEY",
            "OPENAI_COMPAT_API_KEY",
            "OPENAI_API_KEY",
        ],
        "response_format": True,
        "temperature": 0.0,
    },
    "deepseek_r1": {
        "display_name": "DeepSeek-R1",
        "model_name": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "api_key_env": [
            "DEEPSEEK_API_KEY",
            "ZHIZENGZENG_API_KEY",
            "OPENAI_COMPAT_API_KEY",
            "OPENAI_API_KEY",
        ],
        "response_format": False,
        "temperature": 0.6,
    },
    "llama4_scout": {
        "display_name": "Llama4-scout",
        "model_name": "llama-4-scout",
        "base_url": "https://api.zhizengzeng.com/v1/",
        "api_key_env": ["ZHIZENGZENG_API_KEY", "OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"],
        "response_format": False,
        "temperature": 0.0,
    },
}

SETTING_DIRS = {
    "raw_logs_context": "raw_logs_context",
    "static_kb": "static_kb",
    "full_kb": "full_kb",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def clean_json_response(content):
    content = (content or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    match = re.search(r"```json(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    else:
        match_plain = re.search(r"```(.*?)```", content, re.DOTALL)
        if match_plain:
            content = match_plain.group(1).strip()

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(content)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)

    return content


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def resolve_api_key(config, explicit_key):
    if explicit_key:
        return explicit_key
    for env_name in config["api_key_env"]:
        value = os.getenv(env_name)
        if value:
            return value
    names = ", ".join(config["api_key_env"])
    raise RuntimeError(
        f"No API key found. Set one of these environment variables: {names}, "
        "or pass --api-key."
    )


def parse_time(time_full):
    try:
        parts = str(time_full).split()
        return parts[0], parts[1]
    except Exception:
        return "Unknown", "00:00"


def format_raw_logs_context(raw_logs_text):
    text = (raw_logs_text or "").strip()
    text = re.sub(r"^### .*\n", "", text, count=1).strip()
    return "### Recent Raw Historical Activity Memory\n" + text + "\n"


def split_retrieved_context(lines):
    rule_lines = []
    habit_lines = []
    other_lines = []
    for line in lines or []:
        stripped = str(line).strip()
        if stripped.startswith("[Rule "):
            rule_lines.append(stripped)
        elif stripped.startswith("[Habit "):
            habit_lines.append(stripped)
        elif stripped:
            other_lines.append(stripped)
    return rule_lines, habit_lines, other_lines


def format_list_block(lines, empty_text):
    if lines:
        return "\n".join(f"- {line}" for line in lines)
    return empty_text


def format_structured_context(setting, retrieved_context):
    rule_lines, habit_lines, other_lines = split_retrieved_context(retrieved_context)

    if setting == "raw_logs_context":
        return (
            "### Available Structured Knowledge\n"
            "No static rules or long-term habit summaries are provided in this setting.\n"
        )
    if setting == "static_kb":
        static_text = format_list_block(rule_lines + other_lines, "None retrieved for this event.")
        return "### Retrieved Static Rule Context\n" f"{static_text}\n"

    static_text = format_list_block(rule_lines, "None retrieved for this event.")
    habit_text = format_list_block(habit_lines + other_lines, "None retrieved for this event.")
    return (
        "### Retrieved Static Rule Context\n"
        f"{static_text}\n\n"
        "### Retrieved Long-Term Habit Context\n"
        f"{habit_text}\n"
    )


def output_requirements():
    return (
        "Output decision:\n"
        "- Set proactive=true when at least one specific trigger is supported.\n"
        "- Set proactive=false when no specific trigger is supported.\n"
        "- If proactive=true, content must be concise and must name the concrete concern or action.\n"
        "- If proactive=true, decision_basis must briefly state the evidence-based reason for the "
        "intervention. It should identify the trigger type when applicable, such as safety hazard, "
        "anomaly, static-rule violation, short-term deviation, habit omission, habit delay, habit "
        "blockage, habit substitution, habit displacement, or habit conflict.\n"
        "- decision_basis is an evaluation-only grounding field. Do not write a long chain of thought; "
        "write one concise sentence grounded in the current event and available context.\n"
        "- Avoid generic messages such as \"Do you need help?\", \"Is everything okay?\", or \"Would "
        "you like assistance?\" unless tied to a concrete risk, rule violation, anomaly, or deviation.\n\n"
        "Output requirements:\n"
        "- Return a strict JSON object only.\n"
        "- Include the input id exactly.\n"
        "- If proactive=true, content should be a concise user-facing assistance message.\n"
        "- If proactive=true, decision_basis should be a concise grounding statement for the decision.\n"
        "- If proactive=false, content must be an empty string.\n"
        "- If proactive=false, decision_basis must be an empty string.\n\n"
        "JSON format:\n"
        "{\n"
        "  \"id\": \"E001\",\n"
        "  \"proactive\": true,\n"
        "  \"content\": \"Concise assistance message.\",\n"
        "  \"decision_basis\": \"Brief evidence-based reason for the intervention.\"\n"
        "}"
    )


def build_raw_logs_system_prompt():
    return (
        "You are a Proactive Smart Home Assistant.\n"
        "You are in the w/o KB setting. You receive only the current home observation and a short "
        "recent raw historical activity memory. You do not receive static household rules or "
        "long-term habit summaries.\n\n"
        "Decision stance:\n"
        "- Be proactive enough to catch credible needs from the current event, common sense, and "
        "short-term raw memory. Do not be a passive logger.\n"
        "- Be precise, not chatty. A proactive message must name a concrete concern or next action. "
        "If you can only offer a vague check-in, return false.\n"
        "- The current event is the strongest evidence. Recent raw memory is only short-term context; "
        "it is not a rule base and not a long-term habit summary.\n\n"
        "Intervention triggers:\n"
        "1. Safety/common-sense hazards: intervene for falls, fire/flame risk, unattended appliances, "
        "unsafe tool use, health risks, environmental risks, or other credible hazards.\n"
        "2. Anomalous states: intervene when the observation itself is abnormal or concerning, such as "
        "being stationary after a fall, a door left open, an appliance running unattended, or an "
        "unusual state requiring timely checking.\n"
        "3. Short-term deviations from raw memory: intervene for a plausible missed, delayed, blocked, "
        "displaced, or conflicting activity when the current event gives a concrete cue. Useful cues "
        "include still/remains/continues past a relevant time, left/forgot/skipped, unfinished or "
        "unattended state, conflicting location, replacement activity, or explicit absence/delay.\n\n"
        "Non-intervention safeguards:\n"
        "- Return false for ordinary activities, harmless timing variation, normal progress toward a "
        "task, leisure without a concrete issue, or weak speculation.\n"
        "- Do not infer strict personal rules from one or two days of raw logs.\n"
        "- The current event must show risk, omission, delay, blockage, displacement, or conflict.\n\n"
        + output_requirements()
    )


def build_static_kb_system_prompt():
    return (
        "You are a Proactive Smart Home Assistant.\n"
        "You are in the Static KB setting. You receive the current home observation, a short recent "
        "raw historical activity memory, and retrieved static household rules. You do not receive "
        "long-term habit summaries.\n\n"
        "Decision stance:\n"
        "- First judge safety, anomalies, and short-term raw-memory deviations from the current event "
        "and recent raw memory. Then use retrieved static rules as additional explicit constraints.\n"
        "- Static rules expand the reasons to intervene; they must not make you ignore a clear safety "
        "risk, anomaly, or short-term problem already visible from raw evidence.\n"
        "- If no static rule is retrieved, that does not prove the event is normal. Continue using "
        "the current event, common sense, and recent raw memory.\n\n"
        "Intervention triggers:\n"
        "1. Safety/common-sense hazards: intervene for falls, fire/flame risk, unattended appliances, "
        "unsafe tool use, health risks, environmental risks, or other credible hazards.\n"
        "2. Anomalous states: intervene when the observation itself is abnormal or concerning, such as "
        "being stationary after a fall, a door left open, an appliance running unattended, or an "
        "unusual state requiring timely checking.\n"
        "3. Short-term raw-memory deviations: intervene for plausible missed, delayed, blocked, "
        "displaced, or conflicting activity when the current event gives a concrete cue. Do not need "
        "a habit KB for clear short-term problems.\n"
        "4. Static rule violations: intervene when the current event satisfies a retrieved rule's "
        "forbidden, required, deadline, count, duration, or presence condition. For duration/count/"
        "deadline rules, require supporting evidence such as continued activity, elapsed time, missed "
        "deadline, repeated count, or incomplete state. For stove/presence rules, intervene only when "
        "the user is absent, leaving, inattentive, or the appliance/fire is unattended; do not intervene "
        "when the user is clearly present and supervising.\n\n"
        "Non-intervention safeguards:\n"
        "- Return false for ordinary activities, harmless timing variation, normal progress toward a "
        "task, leisure without a concrete issue, or weak speculation.\n"
        "- Do not use raw memory as a long-term habit summary.\n"
        "- Do not trigger a static rule merely because a keyword overlaps; the current event must "
        "semantically satisfy the rule condition.\n\n"
        + output_requirements()
    )


def build_full_kb_system_prompt():
    return (
        "You are a Proactive Smart Home Assistant.\n"
        "You are in the Full KB setting. You receive the current home observation, a short recent raw "
        "historical activity memory, retrieved static household rules, and retrieved long-term habit "
        "context.\n\n"
        "Decision stance:\n"
        "- First judge safety, anomalies, short-term raw-memory deviations, and static rule violations. "
        "Then use retrieved habit context as additional evidence for long-term routine deviations.\n"
        "- Static rules and habit context expand the reasons to intervene; they must not suppress a "
        "clear safety risk, anomaly, rule violation, or short-term problem already visible.\n"
        "- Habit context is probabilistic. A retrieved habit is background evidence, not a reminder list.\n\n"
        "Intervention triggers:\n"
        "1. Safety/common-sense hazards: intervene for falls, fire/flame risk, unattended appliances, "
        "unsafe tool use, health risks, environmental risks, or other credible hazards.\n"
        "2. Anomalous states: intervene when the observation itself is abnormal or concerning, such as "
        "being stationary after a fall, a door left open, an appliance running unattended, or an "
        "unusual state requiring timely checking.\n"
        "3. Static rule violations: intervene when the current event satisfies a retrieved rule's "
        "forbidden, required, deadline, count, duration, or presence condition. For duration/count/"
        "deadline rules, require supporting evidence such as continued activity, elapsed time, missed "
        "deadline, repeated count, or incomplete state. For stove/presence rules, intervene only when "
        "the user is absent, leaving, inattentive, or the appliance/fire is unattended; do not intervene "
        "when the user is clearly present and supervising.\n"
        "4. Short-term raw-memory deviations: intervene for plausible missed, delayed, blocked, "
        "displaced, or conflicting activity when the current event gives a concrete cue.\n"
        "5. Long-term habit deviations: intervene if the current event and retrieved habit context "
        "together indicate omission, delay, overdue status, substitution, blockage, displacement, or "
        "conflict. Retrieved lines such as possibly missed, overdue, blocked, displaced, conflicting "
        "location, or replacement activity are strong supporting evidence when they fit the event.\n\n"
        "Habit safeguards:\n"
        "- A habit match alone is not a reason to intervene.\n"
        "- Treat current routine, nearby routine transition, upcoming routine, or same-location routine "
        "as normal context unless the current event also shows omission, delay, blockage, substitution, "
        "displacement, conflict, or risk.\n"
        "- If the current event is consistent with a retrieved habit or shows normal progress toward it, "
        "remain silent.\n"
        + output_requirements()
    )


def build_system_prompt(setting):
    if setting == "raw_logs_context":
        return build_raw_logs_system_prompt()
    if setting == "static_kb":
        return build_static_kb_system_prompt()
    if setting == "full_kb":
        return build_full_kb_system_prompt()
    raise ValueError(f"Unknown setting: {setting}")


def build_messages(args, item, retriever=None, raw_logs_text=None):
    entry_id = item.get("id", "")
    time_full = item.get("time", "")
    location = item.get("location", "")
    event_desc = item.get("event", "")
    current_day, current_time = parse_time(time_full)

    base_event = (
        f"ID: {entry_id}\n"
        f"Time: {time_full}\n"
        f"Location: {location}\n"
        f"Event Observation: {event_desc}"
    )

    retrieved_context = []
    if retriever is not None:
        retrieved_context = retriever.get_relevant_context(
            current_day,
            current_time,
            location,
            event_desc,
        )

    user_input = (
        f"{format_raw_logs_context(raw_logs_text)}\n"
        f"{format_structured_context(args.setting, retrieved_context)}\n"
        f"### Current Event\n{base_event}"
    )

    return (
        [
            {"role": "system", "content": build_system_prompt(args.setting)},
            {"role": "user", "content": user_input},
        ],
        retrieved_context,
    )


def create_client(args, config):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Python package 'openai' is required for API runs. "
            "Run this script in the conda environment used by the experiment scripts, "
            "or install the package in the current environment."
        ) from exc

    api_key = resolve_api_key(config, args.api_key)
    base_url = args.base_url or config["base_url"]
    return OpenAI(api_key=api_key, base_url=base_url)


def call_model(client, config, args, messages):
    kwargs = {
        "model": args.model_name or config["model_name"],
        "messages": messages,
        "temperature": args.temperature if args.temperature is not None else config["temperature"],
    }
    if config["response_format"] and not args.disable_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if config.get("extra_body"):
        kwargs["extra_body"] = config["extra_body"]
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def is_fatal_api_error(exc):
    text = str(exc).lower()
    fatal_markers = [
        "401",
        "403",
        "access_denied",
        "access denied",
        "unauthorized",
        "permission",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "model not found",
        "model_not_found",
        "does not exist",
    ]
    return any(marker in text for marker in fatal_markers)


def output_path_for(args):
    filename = f"experiment_{args.model}_{args.setting}.json"
    return Path(args.output_root) / SETTING_DIRS[args.setting] / filename


def load_existing_results(path, retry_errors):
    path = Path(path)
    if not path.exists():
        return [], set()
    data = load_json(path)
    done_ids = set()
    for item in data:
        if "id" not in item:
            continue
        if retry_errors and "error" in item:
            continue
        done_ids.add(item["id"])
    if retry_errors:
        data = [item for item in data if not ("error" in item and "id" in item)]
    return data, done_ids


def create_retriever(args):
    if args.setting == "raw_logs_context":
        return None
    if args.setting == "static_kb":
        return KnowledgeRetriever(args.static_kb, None)
    return KnowledgeRetriever(args.static_kb, args.habit_kb)


def normalize_text_field(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def parse_model_decision(raw_content):
    cleaned = clean_json_response(raw_content)
    decision = json.loads(cleaned)
    proactive = coerce_bool(decision.get("proactive", False))
    content = normalize_text_field(decision.get("content", decision.get("proposed_action", "")))
    decision_basis = normalize_text_field(decision.get("decision_basis", ""))
    if not proactive:
        content = ""
        decision_basis = ""
    return decision, proactive, content, decision_basis


def build_result_entry(args, item, proactive, proposed_action, decision_basis):
    return {
        "id": item.get("id"),
        "model": args.model,
        "setting": args.setting,
        "input_time": item.get("time", ""),
        "input_location": item.get("location", ""),
        "input_event": item.get("event", ""),
        "proactive": proactive,
        "proposed_action": proposed_action,
        "decision_basis": decision_basis,
    }


def run_single_experiment(args):
    config = MODEL_CONFIGS[args.model]
    test_data = load_json(args.test_file)
    output_path = output_path_for(args)

    existing_results, done_ids = load_existing_results(output_path, args.retry_errors)
    results = list(existing_results) if args.resume else []
    if not args.resume:
        done_ids = set()

    with open(args.raw_logs_context, "r", encoding="utf-8") as f:
        raw_logs_text = f.read()

    retriever = create_retriever(args)
    client = None if args.dry_run else create_client(args, config)

    print(">>> HomeActEval experiment")
    print(f">>> Model: {config['display_name']} ({args.model})")
    print(f">>> Setting: {args.setting}")
    print(f">>> Test file: {args.test_file}")
    print(f">>> Raw logs context file: {args.raw_logs_context}")
    print(f">>> Output: {output_path}")
    print(f">>> Resume: {args.resume}; already completed: {len(done_ids)}")

    processed_now = 0
    consecutive_errors = 0
    for index, item in enumerate(test_data, start=1):
        entry_id = item.get("id")
        if entry_id in done_ids:
            continue
        if args.limit is not None and processed_now >= args.limit:
            break

        messages, retrieved_context = build_messages(args, item, retriever, raw_logs_text)

        if args.dry_run:
            print("\n=== DRY RUN PROMPT ===")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("=== END DRY RUN ===")
            return

        print(f"[{index}/{len(test_data)}] Processing {entry_id}...", end="", flush=True)
        raw_content = None
        last_error = None
        try:
            raw_content = call_model(client, config, args, messages)
            decision, proactive, proposed_action, decision_basis = parse_model_decision(raw_content)
            result_entry = build_result_entry(
                args=args,
                item=item,
                proactive=proactive,
                proposed_action=proposed_action,
                decision_basis=decision_basis,
            )

            if decision.get("id") not in {None, entry_id}:
                result_entry["model_returned_id"] = decision.get("id")
            if proactive and not decision_basis:
                result_entry["parse_warning"] = "Missing decision_basis for proactive=true."
            if args.save_retrieved_context:
                result_entry["retrieved_context"] = retrieved_context
            if args.save_raw_response:
                result_entry["raw_response"] = raw_content

            results.append(result_entry)
            done_ids.add(entry_id)
            processed_now += 1
            consecutive_errors = 0
            print(f" Done. Proactive: {proactive}")
        except Exception as exc:
            last_error = exc
            result_entry = {
                "id": entry_id,
                "model": args.model,
                "setting": args.setting,
                "input_time": item.get("time", ""),
                "input_location": item.get("location", ""),
                "input_event": item.get("event", ""),
                "decision_basis": "",
                "error": str(exc),
            }
            if args.save_raw_response and raw_content:
                result_entry["raw_response"] = raw_content
            results.append(result_entry)
            done_ids.add(entry_id)
            processed_now += 1
            consecutive_errors += 1
            print(f" Error: {exc}")

        save_json(results, output_path)

        if "error" in results[-1]:
            if last_error is not None and is_fatal_api_error(last_error):
                print(
                    "\nFatal API/access error detected. Stopping early to avoid "
                    "writing the same error for the remaining benchmark items."
                )
                print("After fixing the API key/model access/model name, rerun with --resume --retry-errors.")
                break
            if consecutive_errors >= args.max_consecutive_errors:
                print(
                    f"\nReached {consecutive_errors} consecutive errors. Stopping early. "
                    "Inspect the output before resuming."
                )
                print("After fixing the issue, rerun with --resume --retry-errors.")
                break

    save_json(results, output_path)
    print(f"\nExperiment finished or paused. Results saved to: {output_path}")
    print(f"Current records in output: {len(results)}")


def selected_values(value, all_values):
    if value == "all":
        return list(all_values)
    return [value]


def run_experiment(args):
    requested_models = selected_values(args.model, DEFAULT_MODELS)
    requested_settings = selected_values(args.setting, SETTING_DIRS.keys())
    original_model = args.model
    original_setting = args.setting

    for model in requested_models:
        for setting in requested_settings:
            args.model = model
            args.setting = setting
            run_single_experiment(args)

    args.model = original_model
    args.setting = original_setting


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run HomeActEval proactive-interaction experiments with the released "
            "decision policy and decision_basis output field."
        )
    )
    parser.add_argument("--model", choices=DEFAULT_MODELS + ["all"], required=True)
    parser.add_argument(
        "--setting",
        choices=sorted(SETTING_DIRS.keys()) + ["all"],
        required=True,
        help="Evaluate raw_logs_context, static_kb, and full_kb settings.",
    )
    parser.add_argument("--test-file", default=str(ROOT_DIR / "knowledgeBase" / "test_Bench.json"))
    parser.add_argument("--static-kb", default=str(ROOT_DIR / "knowledgeBase" / "staticKB.json"))
    parser.add_argument(
        "--habit-kb",
        default=str(ROOT_DIR / "knowledgeBase" / "extracted_stable_habits_gpt4o.json"),
    )
    parser.add_argument(
        "--raw-logs-context",
        default=str(
            ROOT_DIR
            / "results"
            / "raw_logs_context"
            / "context_recent_days.txt"
        ),
    )
    parser.add_argument("--output-root", default=str(ROOT_DIR / "results"))
    parser.add_argument("--api-key", default=None, help="API key override. Prefer env vars.")
    parser.add_argument("--base-url", default=None, help="Base URL override.")
    parser.add_argument("--model-name", default=None, help="Provider model name override.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only N new items per run.")
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop after this many consecutive non-fatal errors.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue from an existing output file.")
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess existing error records.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt without API calls.")
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--save-raw-response", action="store_true")
    parser.add_argument("--save-retrieved-context", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
