"""
=============================================================================
Module: Proactive Smart Home Agent Evaluator (Taking DeepSeek-R1 as an example)
Description: 
    This script evaluates the proactive decision-making capabilities of a 
    Smart Home Agent using the DeepSeek-R1 model. It processes simulated 
    daily activity logs and determines whether the agent should intervene 
    based on the user's current context (time, location, event).

Outputs:
    - experiment_deepseek_r1_with_KB.json (or no_KB.json):
      JSON array containing the original event description alongside the 
      agent's boolean decision (`proactive`) and generated action (`proposed_action`).
=============================================================================
"""

import os
import json
import re
from openai import OpenAI
from knowledge_retriever import KnowledgeRetriever 

# ================= Configuration Area =================
API_SECRET_KEY = "sk-xxx"  # your API Key
BASE_URL = "https://api.openai.com/v1"  # your API Base URL


# Input data paths
TEST_FILE = "./test_Bench.json"
STATIC_KB = "./staticKB.json"
HABIT_KB = "./stable_habits.json"
HABIT_KB = None

# Output paths configuration
NO_KB_DIR = "./outputs/no_kb"
WITH_KB_DIR = "./outputs/only_staticKB"

# === Evaluation Mode Setup ===
USE_KB = True  

if USE_KB:
    OUTPUT_DIR = WITH_KB_DIR
    OUTPUT_FILENAME = "experiment_deepseek_r1_with_KB.json"
else:
    OUTPUT_DIR = NO_KB_DIR
    OUTPUT_FILENAME = "experiment_deepseek_r1_no_KB.json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# ======================================================

client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_json_response(content):
    """
    Dedicated parser for DeepSeek-R1 output.
    1. Removes the <think>...</think> reasoning chain.
    2. Extracts the JSON payload from the markdown ```json ... ``` block.
    """
    # 1. Remove Chain-of-Thought content (DeepSeek-R1 specific)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    
    # 2. Attempt to extract the Markdown JSON block
    match = re.search(r"```json(.*?)```", content, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        # Fallback: find the first '{' to the last '}'
        match_brace = re.search(r"(\{.*\})", content, re.DOTALL)
        if match_brace:
            content = match_brace.group(1).strip()
            
    return content

def run_experiment():
    print(f">>> Starting Experiment using [DeepSeek-R1]. USE_KB = {USE_KB}")
    print(f">>> Output will be saved to: {OUTPUT_FILE_PATH}")
    
    test_data = load_json(TEST_FILE)
    
    retriever = None
    if USE_KB:
        retriever = KnowledgeRetriever(STATIC_KB, HABIT_KB)
        print(">>> Knowledge Bases Loaded.")

    results = []
    
    for index, item in enumerate(test_data):
        entry_id = item.get('id')
        time_full = item.get('time')
        location = item.get('location')
        event_desc = item.get('event')
        
        print(f"[{index+1}/{len(test_data)}] Processing {entry_id}...", end="")

        try:
            parts = time_full.split()
            current_day = parts[0]
            current_time = parts[1]
        except:
            current_day = "Unknown"
            current_time = "00:00"

        context_str = "No specific user rules or habits found for this context."
        if USE_KB and retriever:
            retrieved_items = retriever.get_relevant_context(current_day, current_time, location)
            if retrieved_items:
                context_str = "Relevant User Rules/Habits:\n" + "\n".join(f"- {r}" for r in retrieved_items)
        
        # === Build Prompt ===
        system_prompt = (
            "You are a Proactive Smart Home Assistant.\n"
            "Your Goal: Determine if you should intervene (be proactive) based on the user's event, context, and safety/common sense.\n\n"
            "### Input Structure:\n"
            "1. Event details (ID, Time, Location, Observation)\n"
            "2. **Context Information**: A list of retrieved User Rules and Habits relevant to this event.\n\n"
            "### Decision Logic:\n"
            "1. **Safety First (World Knowledge)**: If the user is in danger (falling, fire, injury), ALWAYS match proactive=true, even if no rule exists.\n"
            "2. **Rule/Habit**: If the event violates a known Rule or significantly deviates from a Habit in the Context, proactive=true.\n"
            "3. **Habit Support (Context)**: If the user deviates significantly from a strict habit or needs reminders, consider proactive=true.\n"
            "4. **Silence**: If the event is normal behavior and violates no safety/rules, return proactive=false.\n\n"
            "### Output Requirements:\n"
            "1. **Must include the Input ID** ('id').\n"
            "2. **Content**: If proactive=true, provide the exact sentence/action. If false, empty string.\n"
            "3. **IMPORTANT**: Return ONLY the JSON object. Do not include reasoning outside the JSON.\n\n"
            "### Output Format (Strict JSON):\n"
            "{\n"
            "  \"id\": \"E001\",\n"
            "  \"proactive\": true,\n"
            "  \"content\": \"Warning: Fall detected. Calling emergency contacts immediately.\"\n"
            "}"
        )

        user_input = (
            f"ID: {entry_id}\n"  
            f"Time: {time_full}\n"
            f"Location: {location}\n"
            f"Event Observation: {event_desc}\n\n"
            f"### Context Information (User Rules & Habits):\n{context_str}"
        )

        try:
            # Temperature is set to 0.6 as recommended for R1 to enable better reasoning capabilities.
            # 0.0 may result in repetitive loops.
            response = client.chat.completions.create(
                model="deepseek-r1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.6 
            )
            
            raw_content = response.choices[0].message.content
            
            content = clean_json_response(raw_content)
            llm_decision = json.loads(content)
            
            result_entry = {
                "id": llm_decision.get("id", entry_id),
                "input_event": event_desc,
                "proactive": llm_decision.get("proactive", False),
                "proposed_action": llm_decision.get("content", "")
            }
            results.append(result_entry)
            print(f" Done. Proactive: {result_entry['proactive']}")

            # Save incrementally to prevent data loss on crash
            with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f" Error: {e}")
            error_entry = {"id": entry_id, "error": str(e)}
            results.append(error_entry)
            with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nExperiment Finished. Final results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    run_experiment()