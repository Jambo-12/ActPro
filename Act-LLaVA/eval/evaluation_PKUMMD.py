import json
import os
import glob

def normalize_text(text):
    """标准化：提取内容、转小写、去首尾空格及末尾句号"""
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    return text.strip().lower().rstrip('.')

def calculate_class_wise_metrics(window_size=None):
    results_dir = 'dataset/PKUMMD/results'
    gt_path = 'dataset/PKUMMD/annotations/test.json'
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # 1. 获取所有类别并初始化统计字典
    all_classes = set()
    for vid in gt_data:
        for item in gt_data[vid]:
            all_classes.add(normalize_text(item['text']))
    
    class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_classes}
    
    pred_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for pred_file in pred_files:
        video_id = os.path.basename(pred_file).replace('.json', '')
        if video_id not in gt_data: continue
            
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_content = json.load(f)
            
        video_gt = gt_data[video_id]
        preds = [item for item in pred_content['conversation'] if 'role' in item and item['role'] == 'assistant']
        
        # 用于追踪哪些预测和GT已经被消耗掉
        matched_pred_indices = set()
        matched_gt_indices = set()

        # 2. 匹配 TP (True Positives)
        for g_idx, gt_item in enumerate(video_gt):
            cls = normalize_text(gt_item['text'])
            
            # 计算动态评估区间
            if window_size is None:
                eval_start, eval_end = gt_item['start_time'], gt_item['end_time']
            else:
                eval_start = max(gt_item['start_time'], gt_item['timestamp'] - window_size)
                eval_end = min(gt_item['end_time'], gt_item['timestamp'] + window_size)
            
            for p_idx, p_item in enumerate(preds):
                if p_idx in matched_pred_indices: continue # 每个预测只能被用作一次 TP
                
                # 如果时间落入区间且文本匹配
                if eval_start <= p_item['time'] <= eval_end:
                    if normalize_text(p_item['content']) == cls:
                        class_stats[cls]['tp'] += 1
                        matched_pred_indices.add(p_idx)
                        matched_gt_indices.add(g_idx)
                        break # 一个GT动作找到一个TP即可

        # 3. 统计 FN (False Negatives) - 漏报
        for g_idx, gt_item in enumerate(video_gt):
            if g_idx not in matched_gt_indices:
                cls = normalize_text(gt_item['text'])
                class_stats[cls]['fn'] += 1

        # 4. 统计 FP (False Positives) - 误报/多余响应
        # 凡是没能匹配上 TP 的预测，无论是因为：
        # a) 时间不对  b) 话术不对  c) 话术对了但是是重复说的
        # 全部计入该预测所声称类别的 FP
        for p_idx, p_item in enumerate(preds):
            if p_idx not in matched_pred_indices:
                p_cls = normalize_text(p_item['content'])
                if p_cls in class_stats:
                    class_stats[p_cls]['fp'] += 1
                else:
                    # 如果模型预测了一个 label.json 里根本没有的动作类别，
                    # 可以在这里打印出来调试，或者忽略。
                    pass

    # --- 最终指标计算 ---
    print(f"## 评估报告 (Window Size: {window_size if window_size is not None else 'Full'})")
    print(f"| Action Class | Precision | Recall | F1 | TP/FP/FN |")
    print(f"| :--- | :--- | :--- | :--- | :--- |")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    macro_p_sum, macro_r_sum = 0, 0

    for cls in sorted(all_classes):
        s = class_stats[cls]
        p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        print(f"| {cls} | {p:.4f} | {r:.4f} | {f1:.4f} | {s['tp']}/{s['fp']}/{s['fn']} |")
        
        total_tp += s['tp']
        total_fp += s['fp']
        total_fn += s['fn']
        macro_p_sum += p
        macro_r_sum += r

    num_classes = len(all_classes)
    macro_p, macro_r = macro_p_sum / num_classes, macro_r_sum / num_classes
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print(f"\n### 汇总统计")
    print(f"* **Macro Average**: P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print(f"* **Micro Average**: P={micro_p:.4f}, R={micro_r:.4f}, F1={micro_f1:.4f}")

if __name__ == "__main__":
    # 你可以调整 window_size 来查看不同时间精度下的模型表现
    # None 代表使用 start_time 到 end_time
    # 1.0 代表使用 [timestamp-1.0, timestamp+1.0] 且不超出原始范围
    calculate_class_wise_metrics(window_size=None)