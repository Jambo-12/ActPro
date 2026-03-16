import json
from metrics import ClassificationMetrics
# from word_match import Word_Match # 如果您没用到可以保持注释
import os

class DataProcessor:
    """
    用于解析预测文件和标签文件，并划分为样本的类。
    """

    def __init__(self, delimiter="\n", encoding="utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding
        self.action_keywords = [
            "read", "writ", "stomach", "using a phone", "using a laptop",
            "drink", "pick", "reach", "pour", "eat", "headache", "wash", "carry",
            "taking off", "operat", "putting on", "sit", "stand", "lying"
        ]
    
    def parse_pre_file(self, file_path):
        timestamps, captions = [], []
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                content = json.load(file)
                for item in content.get('conversation', []):
                    if item.get('role') == "assistant":
                        captions.append(item.get('content', ""))
                        timestamps.append(item.get('time', ""))
        except Exception as e:
            print(f"解析文件 {file_path} 出现错误: {e}")
        
        return timestamps, captions

    def parse_label_file(self, file_path):
        timestamps, labels, intervals = [], [], []
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                content = json.load(file)
                assert len(content.keys()) == 1

                for item in list(content.values())[0]:
                    labels.append(item.get('text', ""))
                    timestamps.append(item.get('timestamp', ""))
                    intervals.append((int(item.get('start_time', 0)), int(item.get('end_time', 0))))
        except Exception as e:
            print(f"解析文件 {file_path} 出现错误: {e}")
        
        return timestamps, labels, intervals

    def split_time(self, timestamps, intervals, threshold=2):
        split_intervals = []
        flags = []
        last_time = 0

        for timestamp, interval in zip(timestamps, intervals):
            start_time = max(interval[0], timestamp - threshold)
            end_time = min(interval[1], timestamp + threshold)

            if start_time > last_time: 
                split_intervals.append((last_time, start_time))
                flags.append(0)

            if end_time > start_time:
                split_intervals.append((start_time, end_time))
                flags.append(1)
            
            last_time = end_time
        split_intervals.append((last_time, last_time + 50))
        flags.append(0)

        return split_intervals, flags
    
    def assign_points_to_intervals(self, intervals, points):
        result = [[] for _ in intervals]
        for point in points:
            for i, (start, end) in enumerate(intervals):
                if start <= point < end:
                    result[i].append(point)
                    break
        return result

    def extract_keywords(self, text):
        return {keyword for keyword in self.action_keywords if keyword in text}

    def compare_active(self, segments, space_flags):
        match_result = []
        for seg_id, segment in enumerate(segments):
            if not space_flags[seg_id]:
                match_result.append(0)
            elif len(segment):
                match_result.append(1)
            else:
                match_result.append(0)
        return match_result

    def compare_caption(self, segments, true_labels, predicted_labels, space_flags):
        """
        修改点：同时记录每个区间中提取到的真实类别（true_kws）和预测类别（pred_kws），
        以便后续为每个类别生成单独的二进制序列计算宏平均。
        """
        true_idx, pred_idx = 0, 0
        match_result = []
        true_kws_list = []
        pred_kws_list = []

        for seg_id, segment in enumerate(segments):
            if not space_flags[seg_id]:
                seg_pred_kws = set()
                # 遍历处理所有的预测，收集假阳性（FP）的关键词并防止 pred_idx 错位
                for _ in range(len(segment)):
                    seg_pred_kws.update(self.extract_keywords(predicted_labels[pred_idx]))
                    pred_idx += 1
                match_result.append(0)
                true_kws_list.append(set())
                pred_kws_list.append(seg_pred_kws)
            else:
                matched = 0
                true_keywords = self.extract_keywords(true_labels[true_idx])
                seg_pred_kws = set()
                for _ in range(len(segment)):
                    pred_keywords = self.extract_keywords(predicted_labels[pred_idx])
                    seg_pred_kws.update(pred_keywords)
                    if true_keywords == pred_keywords:
                        matched = 1
                    pred_idx += 1
                
                true_idx += 1
                match_result.append(matched)
                true_kws_list.append(true_keywords)
                pred_kws_list.append(seg_pred_kws)
        
        return match_result, true_kws_list, pred_kws_list

    def binary_data(self, interval, label_flag, pred_points, pred_flag):
        time_len = int((interval[1] - interval[0]) * 2) + 1
        label = [0] * time_len
        predict = [0] * time_len
        pred_positions = [int((pt - interval[0])) * 2 for pt in pred_points]

        for idx in pred_positions:
            predict[idx] = 1

        if label_flag:
            if pred_flag:
                if pred_positions:
                    label[pred_positions[0]] = 1
                else:
                    label[0] = 1
            elif pred_positions:
                label[pred_positions[0]] = 1
                predict[pred_positions[0]] = 0
            else:
                label[0] = 1

        return label, predict

    def binary_set(self, intervals, flags, pred_points, pred_flags):
        labels, predictions = [], []
        for i in range(len(intervals)):
            lbl, pred = self.binary_data(intervals[i], flags[i], pred_points[i], pred_flags[i])
            labels.extend(lbl)
            predictions.extend(pred)
        return labels, predictions

    def process_files(self, pred_file_path, label_file_path):
        pred_times, pred_caps = self.parse_pre_file(pred_file_path)
        label_times, label_caps, label_intervals = self.parse_label_file(label_file_path)
        
        split_intervals, flags = self.split_time(label_times, label_intervals)
        points_in_intervals = self.assign_points_to_intervals(split_intervals, pred_times)

        # 接收新增的 true_kws_list 和 pred_kws_list
        pred_flags, true_kws_list, pred_kws_list = self.compare_caption(points_in_intervals, label_caps, pred_caps, flags)
        labels, predictions = self.binary_set(split_intervals, flags, points_in_intervals, pred_flags)

        # 修改点：为每个特定的 class 生成二进制 label 和 prediction 序列
        class_labels = {kw: [] for kw in self.action_keywords}
        class_preds = {kw: [] for kw in self.action_keywords}
        
        for i in range(len(split_intervals)):
            interval = split_intervals[i]
            pred_points = points_in_intervals[i]
            true_kws = true_kws_list[i]
            pred_kws = pred_kws_list[i]
            
            time_len = int((interval[1] - interval[0]) * 2) + 1
            pred_positions = [int((pt - interval[0])) * 2 for pt in pred_points]
            
            for kw in self.action_keywords:
                lbl = [0] * time_len
                prd = [0] * time_len
                
                kw_label_flag = kw in true_kws
                kw_pred_flag = kw in pred_kws
                
                # 同步原有的时序摆放逻辑
                for idx in pred_positions:
                    if kw_pred_flag: 
                        prd[idx] = 1
                        
                if kw_label_flag:
                    if kw_pred_flag and pred_positions:
                        lbl[pred_positions[0]] = 1
                    elif pred_positions:
                        lbl[pred_positions[0]] = 1
                        prd[pred_positions[0]] = 0 
                    else:
                        lbl[0] = 1
                        
                class_labels[kw].extend(lbl)
                class_preds[kw].extend(prd)

        # 返回整体指标以及按类别拆分的指标
        return labels, predictions, class_labels, class_preds

def find_file(folder_path):
    file_list = os.listdir(folder_path)
    final_list = []
    for file_name in file_list:
        if file_name.split('.')[1] == 'json':
            final_list.append(file_name)
    return final_list

if __name__ == '__main__':
    pre_folder = '/work/pqz/dataset/xdu_2fps/test/experiment/test_8/stepall'
    lab_folder = '/rest/jambo/Datasets/timestampData/Dataset_info/annotations/test'
    processor = DataProcessor()
    
    name_list = find_file(pre_folder)
    
    # 全局容器
    label_dataset = []
    pre_dataset = []
    global_class_labels = {kw: [] for kw in processor.action_keywords}
    global_class_preds = {kw: [] for kw in processor.action_keywords}

    for name in name_list:
        pre_file = os.path.join(pre_folder, name)
        lab_file = os.path.join(lab_folder, name)
        
        # 接收分解后的多分类数据
        labels, predictions, class_labels, class_preds = processor.process_files(pre_file, lab_file)
        
        label_dataset.extend(labels)
        pre_dataset.extend(predictions)
        
        # 聚合每个类别的数据
        for kw in processor.action_keywords:
            global_class_labels[kw].extend(class_labels[kw])
            global_class_preds[kw].extend(class_preds[kw])
            
        metric = ClassificationMetrics(labels, predictions)
        output = metric.get_all_metrics()
        print(name, ': ', output)
        
    print("\n" + "="*50)
    print("************ 总体指标 (Micro Average / Overall) ************")
    metric = ClassificationMetrics(label_dataset, pre_dataset)
    output = metric.get_all_metrics()
    print(output)
    
    print("\n" + "="*50)
    print("************ 类别宏平均指标 (Macro Average) ************")
    macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
    valid_classes = 0 # 用于避免无数据的类别拉低平均分
    
    for kw in processor.action_keywords:
        c_labels = global_class_labels[kw]
        c_preds = global_class_preds[kw]
        
        # 手动计算每个类别的 TP, FP, FN
        tp = sum(1 for l, p in zip(c_labels, c_preds) if l == 1 and p == 1)
        fp = sum(1 for l, p in zip(c_labels, c_preds) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(c_labels, c_preds) if l == 1 and p == 0)
        
        # 只有当该类别在数据集（或预测）中实际出现过，才纳入宏平均计算
        if (tp + fp + fn) > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            valid_classes += 1
            
            print(f"[{kw:<15}] P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | (TP:{tp}, FP:{fp}, FN:{fn})")

    if valid_classes > 0:
        macro_precision /= valid_classes
        macro_recall /= valid_classes
        macro_f1 /= valid_classes
        
    print("-" * 50)
    print(f"Macro Average  -> Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")