'''
    =============================================================================
    Script Name: Proactive Effectiveness Assessment: (LLM-as-a-Judge)
    Description: 
        This script utilizes a Large Language Model (GPT-4o) as an automated judge 
        to evaluate the quality of a smart home agent's proactive interventions 
        (proposed actions) in response to specific user events.

    Evaluation Modes (EVALUATION_MODE):
    1. "WITH_KB": Assumes the Agent HAD access to the Knowledge Bases. 
                  Evaluates whether it strictly followed explicit rules and user habits.
    2. "NO_KB":   Assumes the Agent had NO access to the KBs and relied on common sense.
                  The KBs are used strictly by the judge as the objective "Ground Truth" 
                  to penalize hallucinations or physically impossible actions.

    - OUTPUT_FILE (Scored Results): 
        Format: JSON list. Appends an "evaluation" object to each original input item.
'''
import json
import os
from openai import OpenAI
from tqdm import tqdm

# ================= KB settings =================

EVALUATION_MODE = "NO_KB"  # "WITH_KB" or "NO_KB"

# 输入文件路径
INPUT_FILE = "./experiment_deepseek_v3_no_KB.json"
# 知识库路径
STATIC_KB_PATH = "./knowledgeBase/staticKB.json"
HABIT_KB_PATH = ".  /knowledgeBase/final_stable_habits_v2.json" # use the final stable habits KB after cleaning and merging

# 输出文件路径
OUTPUT_FILE = "./outputs/quality_score/experiment_deepseek_v3_no_KB_SCORED.json"

API_KEY = "sk-xxx"  # your API Key
BASE_URL = "https://api.openai.com/v1"  # your API Base URL

# ===========================================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return [] 

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def get_with_kb_prompt(event, action, static_kb, habit_kb):
    return f"""
You are a highly critical and evaluator for a smart home AI agent. 
The Agent **HAS ACCESS** to the specific Knowledge Bases (KBs) provided below. 
Your goal is to assess the quality of the agent's proactive intervention.

### Reference Knowledge Bases (The Agent HAD access to these):
1. **Static Knowledge Base (Explicit Rules & Device List):**
{json.dumps(static_kb, indent=2, ensure_ascii=False)}

2. **User Habits Knowledge Base (Strict Preferences):**
{json.dumps(habit_kb, indent=2, ensure_ascii=False)}

### Scenario:
**Input Event:** "{event}"
**Agent's Proposed Action:** "{action}"

### Evaluation Mode: 
- **SCORING ANCHOR:**
    - **90-100 (Perfect):** The action perfectly cites or aligns with a specific KB Rule AND respects all User Habits.
    - **65-90 (Mediocre):** The action is generally safe but fails to explicitly utilize the specific logic/devices in the KB.
    - **< 60 (Fail):** The action contradicts a KB rule or habit.
- **ZERO TOLERANCE FOR HALLUCINATIONS:** - If the agent mentions a device NOT present in the Static KB -> **Correctness = 0**.
    - If the agent invents a rule NOT in the Static KB -> **Correctness = 0**.

### Evaluation Criteria (Mode: WITH KB):
1. **Correctness (0-100):** - Did the agent strictly follow the logic of the **Static KB rules** (e.g., if Rule R010 says "Turn off stove if no one is present", did the agent follow it)? 

2. **Contextual Alignment (0-100):** - Did the agent cross-reference the **User Habits KB**? 
   - Does the action match the user's preference for the current time/activity?

3. **Clarity (0-100):** - Is the output concise, polite, and easy to understand for the user?

### Output Format (JSON Only):
Provide your reasoning FIRST.
{{
  "reasoning": "Step 1: Check Static KB compliance (cite specific Rule IDs if applicable). Step 2: Check Habit KB compliance. Step 3: Identify any hallucinations...",
  "correctness": <int>,
  "contextual_alignment": <int>,
  "clarity": <int>
}}
"""



def get_no_kb_prompt(event, action, static_kb, habit_kb):
    return f"""
You are an objective judge evaluating a smart home Agent that had **NO ACCESS** to any external Knowledge Bases. 
The Agent relied solely on its internal Large Language Model (LLM) common sense.

### Ground Truth (For YOUR reference only - The Agent did NOT see this):
Use these KBs only to verify if the Agent's action is physically possible (no hallucinations of devices) and objectively safe.
1. **Static Knowledge Base (Ground Truth of Environment):**
{json.dumps(static_kb, indent=2, ensure_ascii=False)}
2. **User Habits (Ground Truth of Preferences):**
{json.dumps(habit_kb, indent=2, ensure_ascii=False)}

### Scenario:
**Input Event:** "{event}"
**Agent's Proposed Action:** "{action}"

### Evaluation Mode: COMMON SENSE & SAFETY (NO KB)
- **SCORING ANCHOR:**
    - **90-100 (Perfect Common Sense):** The action is objectively safe, logical, and helpful. Crucially, it **coincidentally aligns** with the Ground Truth reality (e.g., suggests turning off a stove, and the house actually has a stove).
    - **65-90 (Safe but Generic):** The action is safe but vague (e.g., "Please check the appliance" instead of "Turn off the stove") or slightly misaligned with the specific user context but acceptable.
    - **< 60 (Fail):** The action is unsafe, annoying (e.g., waking a sleeping user for a trivial matter), or physically impossible based on the Ground Truth.
- **CRITICAL CONSTRAINT - NO "FAKE" COMPLIANCE:**
    - **Do NOT** give high scores because the agent "followed Rule R010". The agent does NOT know Rule R010.
    - If the action matches a rule, credit it as **"Successful Application of Common Sense"**, NOT "Rule Adherence".
- **ZERO TOLERANCE FOR SPECIFIC HALLUCINATIONS:**
    - If the agent invents a **specific** device that does not exist in the Ground Truth (e.g., "Activating the hallway sprinkler system" when none exists) -> **Correctness = 0**.
    - (Note: Generic terms like "smart light" are acceptable if a light exists, but specific invented capabilities are not).

### Evaluation Criteria (Mode: NO KB):
1. **Correctness (0-100):** Is the action physically possible in this specific home (based on Ground Truth KB) and objectively safe?
2. **Contextual Alignment (0-100):** Does the action make sense for the generic situation? (Penalize if it violates a Ground Truth habit that any reasonable person would guess, e.g., don't shout when someone is sleeping).
3. **Clarity (0-100):** Is the message clear and polite?

### Output Format (JSON Only):
Provide reasoning FIRST.
{{
  "reasoning": "Evaluate based on general logic. Explicitly state that the agent acted without KB knowledge. Verify if the action is physically possible in the Ground Truth environment...",
  "correctness": <int>,
  "contextual_alignment": <int>,
  "clarity": <int>
}}
"""



def evaluate_response(event, action, static_kb, habit_kb):
    if EVALUATION_MODE == "WITH_KB":
        prompt = get_with_kb_prompt(event, action, static_kb, habit_kb)
    else:
        prompt = get_no_kb_prompt(event, action, static_kb, habit_kb)
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fair judge. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")
        return None

def main():
    print(f"Starting Evaluation in [{EVALUATION_MODE}] mode...")
    
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    input_data = load_json(INPUT_FILE)
    static_kb = load_json(STATIC_KB_PATH)
    habit_kb = load_json(HABIT_KB_PATH)

    existing_output = load_json(OUTPUT_FILE)
    evaluated_ids = {item['id'] for item in existing_output if 'id' in item}
    final_results = existing_output

    print(f"Loaded {len(input_data)} input items.")
    print(f"Found {len(final_results)} already evaluated items.")

    items_to_process = [
        item for item in input_data 
        if item.get("proactive") is True and item.get("id") not in evaluated_ids
    ]
    
    print(f"Items remaining to evaluate: {len(items_to_process)}")

    if not items_to_process:
        print("All proactive items have been evaluated.")
        return

    for item in tqdm(items_to_process, desc="Evaluating"):
        event = item.get("input_event")
        action = item.get("proposed_action")
        
        scores = evaluate_response(event, action, static_kb, habit_kb)
        
        if scores:
            result_item = item.copy()
            result_item["evaluation"] = scores
            
            avg_score = (scores['correctness'] + scores['contextual_alignment'] + scores['clarity']) / 3
            result_item["evaluation"]["average_score"] = round(avg_score, 2)
            
            result_item["evaluation"]["eval_mode"] = EVALUATION_MODE 
            
            final_results.append(result_item)
            save_json(final_results, OUTPUT_FILE)
        else:
            print(f"Skipping item {item.get('id')} due to API error.")

    print(f"\nDone! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()