"""
=============================================================================
Script Name: Long-Term Habit Induction Pipeline
Description: 
    This script processes long-term chronological activity logs to automatically identify and extract a user's stable 
    daily and weekly routines using an LLM (GPT-4o).

=============================================================================
"""

import os
import json
import time
import re
from openai import OpenAI
from collections import defaultdict

API_SECRET_KEY = "sk-xxx"  
BASE_URL = "https://api.openai.com/v1"  

client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

# ==========================================
# 1. Basic Utilities
# ==========================================

def load_all_data(base_path):
    all_days = []
    for i in range(1, 4): 
        file_path = os.path.join(base_path, f"month_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = data.get("records", [])
                all_days.extend(records)
    return all_days

def clean_json_response(content):
    content = content.strip()
    match = re.search(r"```json(.*?)```", content, re.DOTALL)
    if match:
        content = match.group(1).strip()
    return content

# ==========================================
# 2. Phase 1: Intelligent Scanning (with Exclusion List)
# ==========================================

def get_raw_observations(window_data, mode="daily", exclude_habits=None):
    if not window_data:
        return []

    def clean_action(action_str):
        return action_str.replace("User is ", "").replace("User ", "").strip()

    # --- Branch A: Daily Mode ---
    if mode == "daily":
        cleaned_logs = []
        for day in window_data:
            simple_day = {
                "day": day.get("day_of_week", "Unknown"),
                "date": day.get("date", "Unknown"),
                "events": [
                    {
                        "t": e.get("time"), 
                        "l": e.get("location"), 
                        "a": clean_action(e.get("action"))
                    } 
                    for e in day.get("events", [])
                ]
            }
            cleaned_logs.append(simple_day)
        
        input_data_str = json.dumps(cleaned_logs, ensure_ascii=False)
        
        system_prompt = (
            "You are a 'Habit Profiler'. Analyze the chronological logs.\n"
            "Goal: Identify **Stable Daily Routines** (actions occurring daily).\n"
            "Criteria:\n"
            "1. Temporal Consistency: Same time range daily.\n"
            "2. Contextual Consistency: Same location/sequence.\n"
            "3. Semantic Grouping: 'Eating sandwich' and 'Eating rice' are both 'Lunch'.\n"
            "Return JSON: { \"candidates\": [ { \"desc\": \"...\", \"time\": \"...\", \"confidence\": \"High\" } ] }"
        )
        
        return call_llm(system_prompt, input_data_str)

    # --- Branch B: Weekly Mode ---
    else: 
        grouped_data = defaultdict(list)
        for day in window_data:
            w_name = day.get('day_of_week', 'Unknown')
            
            day_events = []
            for e in day.get("events", []):
                t = e.get('time')
                act = clean_action(e.get('action'))
                loc = e.get('location', 'Unknown')
                day_events.append(f"[{t} @ {loc}] {act}")
            
            grouped_data[w_name].append({
                "date": day.get("date"),
                "events": day_events
            })

        all_weekly_candidates = []
        target_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        exclusion_prompt_part = ""
        if exclude_habits:
            exclusion_str = ", ".join(exclude_habits[:50]) 
            exclusion_prompt_part = (
                f"\n### EXCLUSION LIST (KNOWN DAILY HABITS):\n"
                f"The following are CONFIRMED Daily Habits: [{exclusion_str}].\n"
                f"**DO NOT** report these as weekly habits. Ignore them completely.\n"
            )
        
        print(f"\n    > Deep Scanning Weekdays:", end="")
        
        for day_name in target_weekdays:
            if day_name not in grouped_data:
                continue
                
            print(f" {day_name}...", end="", flush=True)
            
            single_day_data = grouped_data[day_name]
            input_json_str = json.dumps({ day_name: single_day_data }, ensure_ascii=False, indent=2)
            
            system_prompt = (
                f"You are analyzing specific habits for **{day_name}**.\n"
                f"Input: A list of logs from multiple {day_name}s over the past month.\n"
                f"{exclusion_prompt_part}\n" 
                "### YOUR TASK:\n"
                f"Identify actions that repeat across these {day_name}s but are NOT in the Exclusion List.\n\n"
                
                "### CRITICAL DETECTION RULES:\n"
                "1. **Subtract Daily Noise**: If actions like 'Sleep', 'Eat', 'Shower' match the Exclusion List, IGNORE them.\n"
                f"2. **Find the Signal**: Look for '{day_name} Specials' (e.g., 'Weekly Meeting', 'House Cleaning', 'Gaming', 'Yoga').\n"
                "3. **Handle Long Durations (IMPORTANT)**: \n"
                "   - Some habits last a long time (e.g., 'Gaming' from 22:00 to 00:00).\n"
                "   - **Do not require exact timestamp matching**.\n"
                "   - If 'Gaming' appears at 22:10 one week and 22:50 another week, **GROUP THEM** into a single session.\n"
                "   - Use broad time windows in output (e.g., '22:00-00:00' or 'Late Night').\n"
                "4. **Frequency**: The habit session must occur in at least 50% of the provided dates.\n\n"
                
                "### OUTPUT JSON:\n"
                "{ \"candidates\": [ { \"desc\": \"...\", \"period\": \"" + day_name + "\", \"time\": \"...\", \"confidence\": \"High\" } ] }"
            )
            
            candidates = call_llm(system_prompt, input_json_str)
            all_weekly_candidates.extend(candidates)
            
        return all_weekly_candidates
    

def call_llm(system_prompt, user_content):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        content = clean_json_response(response.choices[0].message.content)
        return json.loads(content).get("candidates", [])
    except Exception as e:
        print(f" [LLM Error] {e}")
        return []


# ==========================================
# 3. Phase 2: Global Synthesis
# ==========================================

def synthesize_final_habits(raw_candidates_list):
    candidates_text = ""
    for idx, item in enumerate(raw_candidates_list):
        mini_item = {
            "desc": item.get('desc'),
            "time": item.get('time'),
            "period": item.get('period', 'Daily'), 
            "source": item.get('source_type', 'daily_scan')
        }
        candidates_text += f"{json.dumps(mini_item)}\n"

    print(f"\n\n>>> Phase 2: Synthesizing {len(raw_candidates_list)} candidates into stable habits...")

    system_prompt = (
        "You are the Lead Behavioral Analyst. Merge raw observations into 'Stable Habits'.\n\n"
        "### CONFLICT RESOLUTION (CRITICAL):\n"
        "1. **Weekly Specificity**: If source='weekly_scan' provides a specific period (e.g., 'Tuesday'), prioritize it over a 'Daily' label unless the action happens on >4 different weekdays.\n"
        "2. **Deduplication**: 'Yoga' on Tuesday (from Weekly scan) and 'Exercise' on Tuesday (from Daily scan) are likely the same. Keep the more specific one (Yoga, Weekly).\n"
        "3. **Threshold**: Daily habits must appear frequently. Weekly habits must appear consistently on their specific day.\n\n"
        "### OUTPUT JSON:\n"
        "{\n"
        "  \"stable_rules\": [\n"
        "    {\n"
        "      \"rule_id\": \"H001\",\n"
        "      \"habit_type\": \"daily_routine\" OR \"weekly_routine\",\n"
        "      \"period\": [\"Daily\"] OR [\"Monday\", \"Wednesday\"],\n"
        "      \"description\": \"...\",\n"
        "      \"trigger\": { \"time_window\": \"...\" },\n"
        "      \"confidence\": \"High\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    
    return call_llm(system_prompt, f"Raw Observations:\n{candidates_text}")

# ==========================================
# Main Process Pipeline
# ==========================================

def main_process():
    base_path = "./HomeActEval/HomeStream-6months/"
    
    print(">>> Loading Data...")
    all_days = load_all_data(base_path)
    
    if not all_days:
        print("ERROR: No data loaded.")
        return

    all_candidates = []
    
    # --- Step 1: Daily Scan (Establish Baseline) ---
    print("\n>>> [Phase 1.1] Scanning for Daily candidates...")
    daily_habits_found = [] 
    
    step = 7
    window_size = 14

    for i in range(0, len(all_days), step):
        window = all_days[i : i + window_size]
        if len(window) < 5: break 
        
        print(f"  Scanning Days {i} to {i+len(window)}...", end="")
        res = get_raw_observations(window, mode="daily")
        
        for r in res: 
            r['source_type'] = 'daily_scan'
            if 'desc' in r and r['desc']:
                daily_habits_found.append(r['desc'].strip())
            
        print(f" Found {len(res)}")
        all_candidates.extend(res)
        time.sleep(0.5) 

    confirmed_daily_habits = list(set(daily_habits_found))
    print(f"  > Identified {len(confirmed_daily_habits)} potential daily routines to exclude from weekly scan.")

    # --- Step 2: Weekly Scan (With Exclusion List) ---
    print("\n>>> [Phase 1.2] Scanning for Weekly candidates (With Exclusion)...")
    weekly_step = 14 
    weekly_window = 28
    
    for i in range(0, len(all_days), weekly_step):
        window = all_days[i : i + weekly_window]
        if len(window) < 14: break
        
        print(f"\n  Scanning Window Days {i} to {i+len(window)}...", end="")
        
        res = get_raw_observations(window, mode="weekly", exclude_habits=confirmed_daily_habits)
        
        for r in res: r['source_type'] = 'weekly_scan'
        print(f" -> Found Total {len(res)} candidates.")
        all_candidates.extend(res)

    # --- Step 3: Global Synthesis ---
    if all_candidates:
        candidates_text = ""
        for item in all_candidates:
            mini_item = {
                "desc": item.get('desc'),
                "time": item.get('time'),
                "period": item.get('period', 'Daily'), 
                "source": item.get('source_type', 'daily_scan')
            }
            candidates_text += f"{json.dumps(mini_item)}\n"
            
        print(f"\n\n>>> [Phase 2] Synthesizing {len(all_candidates)} candidates...")
        
        system_prompt = (
            "You are the Lead Behavioral Analyst. Merge raw observations into 'Stable Habits'.\n\n"
            "### CONFLICT RESOLUTION (CRITICAL):\n"
            "1. **Weekly Specificity**: If source='weekly_scan' provides a specific period (e.g., 'Tuesday'), prioritize it over a 'Daily' label unless the action happens on >4 different weekdays.\n"
            "2. **Deduplication**: 'Yoga' on Tuesday (from Weekly scan) and 'Exercise' on Tuesday (from Daily scan) are likely the same. Keep the more specific one (Yoga, Weekly).\n"
            "3. **Threshold**: Daily habits must appear frequently. Weekly habits must appear consistently on their specific day.\n\n"
            "### OUTPUT JSON:\n"
            "{\n"
            "  \"stable_rules\": [\n"
            "    {\n"
            "      \"rule_id\": \"H001\",\n"
            "      \"habit_type\": \"daily_routine\" OR \"weekly_routine\",\n"
            "      \"period\": [\"Daily\"] OR [\"Monday\", \"Wednesday\"],\n"
            "      \"description\": \"...\",\n"
            "      \"trigger\": { \"time_window\": \"...\" },\n"
            "      \"confidence\": \"High\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Raw Observations:\n{candidates_text}"}
                ],
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            content = clean_json_response(response.choices[0].message.content)
            final_habits = json.loads(content).get("stable_rules", [])
        except Exception as e:
            print(f"Synthesis Error: {e}")
            final_habits = []
        
        output_file = "./stable_habits.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_habits, f, indent=2, ensure_ascii=False)
            
        print(f"\nSUCCESS! Extracted {len(final_habits)} stable habits.")
        for h in final_habits:
            period_str = ",".join(h['period']) if isinstance(h['period'], list) else h['period']
            print(f" - [{h.get('habit_type', 'unknown')[:1].upper()}] {h.get('description', '')} ({period_str} @ {h.get('trigger', {}).get('time_window', '')})")
    else:
        print("No candidates found.")

if __name__ == '__main__':
    main_process()