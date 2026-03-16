import os
import json
import numpy as np
from collections import defaultdict

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_by_category(pred_dir, gt_dir, threshold=None):
    """
    threshold: 如果为数字（如 2.0），则区间变为 [timestamp-2, timestamp+2] 与 [start, end] 的交集。
    """
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    
    for filename in pred_files:
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path): continue
            
        with open(pred_path, 'r', encoding='utf-8') as f:
            preds = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        video_id = filename.replace('.json', '')
        gts = gt_data.get(video_id, []) if isinstance(gt_data, dict) else gt_data

        # 预计算每个 GT 的实际判定区间
        effective_intervals = []
        for gt in gts:
            s_orig, e_orig = gt['start_time'], gt['end_time']
            ts = gt.get('timestamp', (s_orig + e_orig) / 2) # 若无timestamp则取中点
            
            if threshold is not None:
                calc_start = max(s_orig, ts - threshold)
                calc_end = min(e_orig, ts + threshold)
            else:
                calc_start, calc_end = s_orig, e_orig
            
            effective_intervals.append({
                'start': calc_start,
                'end': calc_end,
                'text': gt['text'].strip()
            })

        gt_hit_flag = [False] * len(effective_intervals)
        
        # 1. 判定 TP / FP (基于预测)
        for p in preds:
            p_time = p.get('time', 0.0)
            p_text = p.get('content', "").replace("Assistant: ", "").strip()
            
            matched_gt_idx = -1
            for idx, interval in enumerate(effective_intervals):
                if interval['start'] <= p_time <= interval['end']:
                    if p_text == interval['text']:
                        matched_gt_idx = idx
                        break
            
            if matched_gt_idx != -1:
                category_stats[p_text]['tp'] += 1
                gt_hit_flag[matched_gt_idx] = True
            else:
                # 落在区间外或文字错误
                category_stats[p_text]['fp'] += 1
                
        # 2. 判定 FN (基于 GT)
        for idx, interval in enumerate(effective_intervals):
            if not gt_hit_flag[idx]:
                category_stats[interval['text']]['fn'] += 1

    # --- 计算平均值 ---
    cat_precisions, cat_recalls, cat_f1s = [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for cat, stats in category_stats.items():
        p, r, f1 = calculate_metrics(stats['tp'], stats['fp'], stats['fn'])
        cat_precisions.append(p)
        cat_recalls.append(r)
        cat_f1s.append(f1)
        total_tp += stats['tp']
        total_fp += stats['fp']
        total_fn += stats['fn']

    # 指标汇总
    return {
        "Micro_Average": calculate_metrics(total_tp, total_fp, total_fn),
        "Macro_Average_by_Category": (np.mean(cat_precisions), np.mean(cat_recalls), np.mean(cat_f1s)),
        "Count": {"Classes": len(category_stats), "Total_TP": total_tp}
    }

# 示例调用
preds_path = "/rest/jambo/Datasets/timestampData/7Month_4s/LLaVA_ov/2_processed_json"
gts_path = '/rest/jambo/Datasets/timestampData/Dataset_info/annotations/test'
res = evaluate_by_category(preds_path, gts_path, threshold=10.0)
print("Evaluation Results:", res)