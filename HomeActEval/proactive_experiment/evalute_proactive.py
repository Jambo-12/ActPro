"""
=============================================================================
Description: 
    This script evaluates the binary classification performance of the Smart 
    Home Agent's proactive decision-making. It compares the model's predicted 
    interventions against a Ground Truth (GT) dataset.

Key Features:
    1. Overall Metrics: Calculates standard Machine Learning metrics including 
       Precision, Recall, F1-Score, and Accuracy.
    2. Protocol II:
       - TP (True Positive)  : Correct-Detection 
       - FP (False Positive) : False-Detection 
       - FN (False Negative) : Missed-Needed 
       - TN (True Negative)  : Non-Response 

=============================================================================
"""
import json
import os
from collections import defaultdict

GT_FILE = "./test_Bench.json"
PRED_FILE = "./experiment_deepseek_r1_with_KB.json"

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def calculate_metrics(tp, fp, tn, fn):
    """根据 TP, FP, TN, FN 计算 P, R, F1"""
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy = (TP + TN) / Total
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return precision, recall, f1, accuracy

def evaluate_proactive():
    print(f"Loading Ground Truth from: {GT_FILE}")
    print(f"Loading Predictions from: {PRED_FILE}")
    
    gt_data = load_json(GT_FILE)
    pred_data = load_json(PRED_FILE)
    
    # {'E001': {'proactive': True, 'category': 'Anomaly Detection'}, ...}
    gt_map = {item['id']: item for item in gt_data}
    

    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    
    cat_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
    
    processed_ids = set()
    missing_ids = []

    for pred_item in pred_data:
        p_id = pred_item.get('id')
        
        if p_id not in gt_map:
            missing_ids.append(p_id)
            continue
            
        processed_ids.add(p_id)
        
        gt_item = gt_map[p_id]
        category = gt_item.get('category', 'Unknown')
        
        y_true = gt_item.get('proactive')  # True / False
        y_pred = pred_item.get('proactive') # True / False
        

        if y_pred is None:
            y_pred = False

        # Aligns with your Protocol II
        if y_true is True and y_pred is True:
            res_type = 'TP' # Correct-Detection
        elif y_true is False and y_pred is True:
            res_type = 'FP' # False-Detection (Unnecessary Interruption)
        elif y_true is True and y_pred is False:
            res_type = 'FN' # Missed-Needed
        elif y_true is False and y_pred is False:
            res_type = 'TN' # Non-Response
            
        stats[res_type] += 1
        cat_stats[category][res_type] += 1

    print("\n" + "="*50)
    print("OVERALL PERFORMANCE METRICS")
    print("="*50)
    
    tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
    p, r, f1, acc = calculate_metrics(tp, fp, tn, fn)
    
    print(f"Total Samples Processed: {len(processed_ids)}")
    print(f"Confusion Matrix:")
    print(f"  TP (Correct-Detection): {tp}")
    print(f"  FP (False-Detection)  : {fp}")
    print(f"  FN (Missed-Needed)    : {fn}")
    print(f"  TN (Non-Response)     : {tn}")
    print("-" * 30)
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Accuracy  : {acc:.4f}")

    print("\n" + "="*50)
    print("METRICS BY CATEGORY")
    print("="*50)
    print(f"{'Category':<25} | {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4} | {'F1':<6} {'Acc':<6}")
    print("-" * 65)
    
    for cat, s in cat_stats.items():
        ctp, cfp, ctn, cfn = s['TP'], s['FP'], s['TN'], s['FN']
        _, _, cf1, cacc = calculate_metrics(ctp, cfp, ctn, cfn)
        print(f"{cat:<25} | {ctp:<4} {cfp:<4} {cfn:<4} {ctn:<4} | {cf1:.4f} {cacc:.4f}")

    if missing_ids:
        print("\nWarning: Some IDs in prediction were not found in GT:", missing_ids)

if __name__ == "__main__":
    if os.path.exists(GT_FILE) and os.path.exists(PRED_FILE):
        evaluate_proactive()
    else:
        print("Error: Files not found. Please check paths.")