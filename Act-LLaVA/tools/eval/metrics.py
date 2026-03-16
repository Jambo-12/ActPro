class ClassificationMetrics:
    """
    分类指标的计算类，包括 TP、FP、FN、TN，以及 Precision（精确率）、Recall（召回率）和 F1-Score 的计算。
    """
    def __init__(self, y_true, y_pred):
        """
        初始化函数，接收真实标签和预测标签。
        :param y_true: list[int] 或 list[bool]，真实标签列表
        :param y_pred: list[int] 或 list[bool]，预测标签列表
        """
        assert len(y_true) == len(y_pred), "y_true 和 y_pred 长度必须一致。"
        self.y_true = y_true
        self.y_pred = y_pred
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self._calculate_tf_metrics()

    def _calculate_tf_metrics(self):
        """
        私有方法：计算 TP、FP、FN、TN。
        """
        for true, pred in zip(self.y_true, self.y_pred):
            if true == 1 and pred == 1:
                self.TP += 1  # 真阳性
            elif true == 0 and pred == 1:
                self.FP += 1  # 假阳性
            elif true == 1 and pred == 0:
                self.FN += 1  # 假阴性
            elif true == 0 and pred == 0:
                self.TN += 1  # 真阴性

    def precision(self):
        """
        计算 Precision（精确率）。
        :return: 精确率（float）
        """
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0

    def recall(self):
        """
        计算 Recall（召回率）。
        :return: 召回率（float）
        """
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
    
    def accuracy(self):
        """
        计算 Accuracy
        :return: 准确率（float）
        """
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)  if (self.TP + self.FP + self.FN + self.TN) > 0 else 0      

    def f1_score(self):
        """
        计算 F1-Score。
        :return: F1-Score（float）
        """
        precision = self.precision()
        recall = self.recall()
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def get_confusion_matrix(self):
        """
        返回混淆矩阵指标（TP, FP, FN, TN）。
        :return: dict，包含每个指标
        """
        return {
            "TP": self.TP,
            "FP": self.FP,
            "FN": self.FN,
            "TN": self.TN
        }

    def get_all_metrics(self):
        """
        返回所有指标，包括混淆矩阵以及 Precision、Recall 和 F1-Score。
        :return: dict，包含所有指标
        """
        return {
            **self.get_confusion_matrix(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "Accuracy": self.accuracy(),
            "F1-Score": self.f1_score()
        }


