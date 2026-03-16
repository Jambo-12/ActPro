'''
检查 JSON 文件中的t1_caption和text字段中的动作短语一致。
'''

class Word_Match:
    def __init__(self):
        self.action_keywords = [
        "read", "writ", "stomachache", "using a phone", "using a laptop",
        "drink", "pick", "reach", "pour", "eat", "headache",
        "wash", "carry", "taking off", "rest", "operat", "pick", "putting on"]
        self.posture_keywords = ["sit", "stand", "lying"]

    def extract_keywords(self, text):
        found = set()
        for keyword in self.action_keywords:
            if keyword in text:
                found.add(keyword)
        return found

if __name__ == "__main__":
    pass