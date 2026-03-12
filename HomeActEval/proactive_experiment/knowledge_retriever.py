"""
=============================================================================
Module: KnowledgeRetriever
Description: 
    This module implements a strict context-aware retrieval system for a 
    smart home agent. It filters and retrieves relevant static rules and 
    user habits based on the user's current spatial and temporal context.
=============================================================================
"""

import json
import os
import re

class KnowledgeRetriever:
    def __init__(self, static_kb_path, habit_kb_path):
        self.static_rules = self._load_json(static_kb_path)
        self.habit_rules = self._load_json(habit_kb_path)

    def _load_json(self, path):
        if path is None:
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

    def _parse_minutes(self, time_str):
        try:
            time_str = time_str.replace('：', ':')
            h, m = map(int, time_str.strip().split(':'))
            return h * 60 + m
        except:
            return -1

    def _is_time_match(self, current_day, current_time_str, rule_window_str):
        rule = rule_window_str.lower().strip()
        c_day = current_day.lower().strip()
        
        if "always" in rule:
            return True

        days_map = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        rule_has_day = any(d in rule for d in days_map) or 'weekdays' in rule or 'weekends' in rule

        if rule_has_day:
            is_day_match = False
            if c_day in rule: 
                is_day_match = True
            elif 'weekdays' in rule and c_day not in ['saturday', 'sunday']: 
                is_day_match = True
            elif 'weekends' in rule and c_day in ['saturday', 'sunday']: 
                is_day_match = True
            
            if not is_day_match: 
                return False 

        time_match = re.search(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', rule)
        if not time_match:
            return True

        start_str, end_str = time_match.groups()
        curr_mins = self._parse_minutes(current_time_str)
        start_mins = self._parse_minutes(start_str)
        end_mins = self._parse_minutes(end_str)

        if curr_mins == -1 or start_mins == -1 or end_mins == -1: 
            return False

        if start_mins <= end_mins:
            return start_mins <= curr_mins <= end_mins
        else:
            return curr_mins >= start_mins or curr_mins <= end_mins

    def _is_location_match(self, current_loc, rule_loc):
        def tokenize(text):
            text = text.lower().replace('_', ' ').replace(',', ' ')
            return set(text.split())

        c_tokens = tokenize(current_loc)
        r_tokens = tokenize(rule_loc)

        global_keywords = {"anywhere", "all", "home", "house", "indoor", "indoors"}
        if not r_tokens.isdisjoint(global_keywords): 
            return True
        
        if r_tokens.issubset(c_tokens):
            return True

        return False

    def get_relevant_context(self, current_day, current_time_str, current_location):
        relevant_info = []
        all_candidates = []
        
        for r in self.static_rules:
            r['src'] = 'Rule'
            all_candidates.append(r)
        for h in self.habit_rules:
            h['src'] = 'Habit'
            all_candidates.append(h)

        for item in all_candidates:
            rule_loc = item.get('location', 'anywhere')
            rule_time = item.get('time_window', 'always')

            if not self._is_location_match(current_location, rule_loc):
                continue
            
            if not self._is_time_match(current_day, current_time_str, rule_time):
                continue

            r_id = item.get('rule_id', 'N/A')
            desc = item.get('description', '')
            src = item.get('src')
            
            info = f"[{src} {r_id}] {desc} (Scope: {rule_time} @ {rule_loc})"
            relevant_info.append(info)
        
        return relevant_info

if __name__ == "__main__":
    print("Retriever updated with Strict Relevance Logic.")