import json
from metrics import ClassificationMetrics
from word_match import Word_Match
import os

class DataProcessor:
    """
    用于解析预测文件和标签文件，并划分为样本的类。
    """

    def __init__(self, delimiter="\n", encoding="utf-8"):
        """
        初始化类的属性。
        
        参数:
        - delimiter (str): 样本之间的分隔符，默认是换行符 '\n'。
        - encoding (str): 文件的编码格式，默认是 "utf-8"。
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.action_keywords = [
            "read", "writ", "stomach", "using a phone", "using a laptop",
            "drink", "pick", "reach", "pour", "eat", "headache", "wash", "carry",
            "taking off", "operat", "putting on", "sit", "stand", "lying"
        ]
    
    def parse_pre_file(self, file_path):
        """
        解析预测文件，提取时间戳和文本描述。

        参数:
        - file_path (str): 预测文件路径。

        返回:
        - tuple: 
            - timestamps (list): 时间戳列表。
            - captions (list): 文本描述列表。
        """
        timestamps, captions = [], []
        
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                content = json.load(file)
                for item in content.get('conversation', []):
                    if item.get('role') == "assistant":
                        captions.append(item.get('content', ""))
                        timestamps.append(item.get('time', ""))
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到！")
        except json.JSONDecodeError:
            print(f"文件 {file_path} 不是有效的 JSON 格式！")
        except Exception as e:
            print(f"解析文件 {file_path} 出现错误: {e}")
        
        return timestamps, captions

    def parse_label_file(self, file_path):
        """
        解析标签文件，提取时间戳、标注文本和时间区间。

        参数:
        - file_path (str): 标签文件路径。

        返回:
        - tuple: 
            - timestamps (list): 标签时间戳列表。
            - labels (list): 标签文本列表。
            - intervals (list[tuple]): 时间区间列表，格式为 (start_time, end_time)。
        """
        timestamps, labels, intervals = [], [], []
        
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                content = json.load(file)
                assert len(content.keys()) == 1

                for item in list(content.values())[0]:
                    labels.append(item.get('text', ""))
                    timestamps.append(item.get('timestamp', ""))
                    intervals.append((int(item.get('start_time', 0)), int(item.get('end_time', 0))))
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到！")
        except json.JSONDecodeError:
            print(f"文件 {file_path} 不是有效的 JSON 格式！")
        except Exception as e:
            print(f"解析文件 {file_path} 出现错误: {e}")
        
        return timestamps, labels, intervals

    def split_time(self, timestamps, intervals, threshold=4):
        """
        根据时间区间划分时间范围，并对应生成标志位。

        参数:
        - timestamps (list[int]): 时间戳列表。
        - intervals (list[tuple[int, int]]): 时间区间列表。
        - threshold (int): 调整时间区间的阈值。

        返回:
        - tuple:
            - split_intervals (list[tuple[int, int]]): 拆分后的时间区间列表。
            - flags (list[int]): 每个时间区间的标记（1 或 0）。
        """
        split_intervals = []
        flags = []
        last_time = 0

        for timestamp, interval in zip(timestamps, intervals):
            start_time = max(interval[0], timestamp - threshold)
            end_time = min(interval[1], timestamp + threshold)

            if start_time > last_time:  # 填补未标记的空白区域
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
        """
        将时间点分配到相应的时间区间中。

        参数:
        - intervals (list[tuple[float, float]]): 时间区间列表。
        - points (list[float]): 时间点列表。

        返回:
        - list[list[float]]: 每个时间区间包含的时间点列表。
        """
        result = [[] for _ in intervals]
        
        for point in points:
            for i, (start, end) in enumerate(intervals):
                if start <= point < end:
                    result[i].append(point)
                    break
        
        return result

    def extract_keywords(self, text):
        """
        从文本中提取动作关键词。

        参数:
        - text (str): 输入的文本。

        返回:
        - set: 提取到的关键词集合。
        """
        return {keyword for keyword in self.action_keywords if keyword in text}

    def compare_active(self, segments, space_flags):
        """
        比较每个区间内的真实标签和预测标签。

        参数:
        - segments (list[list]): 每个时间区间的时间点。
        - space_flags (list[int]): 区间标记。

        返回:
        - list[int]: 每个空间的匹配标记（1 或 0）。
        """
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
        比较每个区间内的真实标签和预测标签。

        参数:
        - segments (list[list]): 每个时间区间的时间点。
        - true_labels (list[str]): 真实标签列表。
        - predicted_labels (list[str]): 预测标签列表。
        - space_flags (list[int]): 区间标记。

        返回:
        - list[int]: 每个空间的匹配标记（1 或 0）。
        """
        true_idx, pred_idx = 0, 0
        match_result = []

        for seg_id, segment in enumerate(segments):
            if not space_flags[seg_id]:
                pred_idx += len(segment)
                match_result.append(0)
            else:
                matched = 0
                for _ in range(len(segment)):
                    true_keywords = self.extract_keywords(true_labels[true_idx])
                    pred_keywords = self.extract_keywords(predicted_labels[pred_idx])
                    if true_keywords == pred_keywords:
                        matched = 1
                    pred_idx += 1
                true_idx += 1
                match_result.append(matched)
        
        return match_result

    def binary_data(self, interval, label_flag, pred_points, pred_flag):
        """
        根据区间生成二值化的标签和预测。

        参数:
        - interval (tuple): 时间区间。
        - label_flag (bool): 是否生成标签。
        - pred_points (list): 预测点。
        - pred_flag (bool): 是否处理预测。

        返回:
        - tuple: 二值化后的标签和预测列表。
        """
        time_len = int((interval[1] - interval[0]) * 2) + 1
        label = [0] * time_len
        predict = [0] * time_len
        pred_positions = [int((pt - interval[0])) * 2 for pt in pred_points]

        for idx in pred_positions:
            predict[idx] = 1

        if label_flag:
            if pred_flag:
                label[pred_positions[0]] = 1
            elif pred_positions:
                label[pred_positions[0]] = 1
                predict[pred_positions[0]] = 0
            else:
                label[0] = 1

        return label, predict

    def binary_set(self, intervals, flags, pred_points, pred_flags):
        """
        对所有时间区间生成二值化数据。

        参数:
        - intervals (list): 时间区间。
        - flags (list): 标记列表。
        - pred_points (list): 预测点列表。
        - pred_flags (list): 预测标记列表。

        返回:
        - tuple: 二值化的标签和预测的完整列表。
        """
        labels, predictions = [], []

        for i in range(len(intervals)):
            lbl, pred = self.binary_data(intervals[i], flags[i], pred_points[i], pred_flags[i])
            labels.extend(lbl)
            predictions.extend(pred)
        
        return labels, predictions

    def process_files(self, pred_file_path, label_file_path):
        """
        解析预测文件和标签文件，并生成样本。

        参数:
        - pred_file_path (str): 预测文件路径。
        - label_file_path (str): 标签文件路径。

        返回:
        - tuple: 二值化的标签和预测列表。
        """
        pred_times, pred_caps = self.parse_pre_file(pred_file_path)
        label_times, label_caps, label_intervals = self.parse_label_file(label_file_path)
        
        split_intervals, flags = self.split_time(label_times, label_intervals)
        points_in_intervals = self.assign_points_to_intervals(split_intervals, pred_times)

        pred_flags = self.compare_caption(points_in_intervals, label_caps, pred_caps, flags)
        # pred_flags = self.compare_active(points_in_intervals, flags)

        labels, predictions = self.binary_set(split_intervals, flags, points_in_intervals, pred_flags)

        return labels, predictions

def find_file(folder_path):
    file_list = os.listdir(folder_path)
    final_list = []
    for file_name in file_list:
        if file_name.split('.')[1] == 'json':
            final_list.append(file_name)
    return final_list

if __name__ == '__main__':
    pre_folder = 'dataset/ASTime/results'
    lab_folder = 'dataset/ASTime/annotations/test'
    processor = DataProcessor()
    
    name_list = find_file(pre_folder)
    label_dataset = []; pre_dataset = []
    for name in name_list:
        pre_file = os.path.join(pre_folder, name)
        lab_file = os.path.join(lab_folder, name)
        
        labels, predictions = processor.process_files(pre_file, lab_file)
        label_dataset.extend(labels)
        pre_dataset.extend(predictions)
        metric = ClassificationMetrics(labels, predictions)
        output = metric.get_all_metrics()
        # confusion = metric.get_confusion_matrix()
        print(name, ': ', output)
        
print("************")
metric = ClassificationMetrics(label_dataset, pre_dataset)
output = metric.get_all_metrics()
print(output)
