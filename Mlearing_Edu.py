import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

class MMMlearing:
    def __init__(self,
                 backbone='RandomForest'
                 ):
        self.backbone = backbone
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.model = RandomForestClassifier()

    def train(self, seed=0):
        np.random.seed(seed)
        dataset = pd.read_csv(self.dataset_path, sep=',', header=None).values
        np.random.shuffle(dataset)
        data, label = dataset[:,:-1],dataset[:,-1]
        train_index = int((1-self.test_size) * len(dataset))
        train_data, train_label = data[:train_index, :], label[:train_index]
        self.test_set = {
            'data':data[train_index:, :],
            'label':label[train_index:]
        }
        self.model.fit(train_data, train_label)

    def inference(self):
        pred = self.model.predict(self.test_set['data'])
        acc = accuracy_score(self.test_set['label'], pred)
        print('准确率为：{}%'.format(acc*100))

    def load_dataset(self, path, test_size=0.2):
        self.dataset_path = path
        self.test_size = test_size