import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB as gauss


class GaussianNB:
    def __init__(self
                 ):
        self.cwd = os.path.dirname(os.getcwd())  # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.model = gauss()
        self.dataset_path = ' '
        self.test_size = ' '
        self.test_set = ' '


    def train(self, seed=0, data_type='csv'):
        np.random.seed(seed)
        if data_type == 'csv':
            dataset = pd.read_csv(self.dataset_path, sep=',', header=None).values
        np.random.shuffle(dataset)

        data, label = dataset[:, :-1], dataset[:, -1]
        train_index = int((1 - self.test_size) * len(dataset))
        train_data, train_label = data[:train_index, ], label[:train_index]
        self.test_set = {
            'data': data[train_index:, ],
            'label': label[train_index:]
        }
        self.model.fit(train_data, train_label)
        return self.model


    def inference(self, mode='cls'):
        pred = self.model.predict(self.test_set['data'])
        if mode == 'cls':
            acc = accuracy_score(self.test_set['label'], pred)
            print('准确率为：{}%'.format(acc * 100))
        elif mode == 'reg':
            loss = mean_squared_error(self.test_set['label'], pred)
            print('Loss: {}'.format(loss))


    def load_dataset(self, path, test_size=0.2):
        self.dataset_path = path
        self.test_size = test_size


    def save(self):
        print("Saving model checkpoints...")
        joblib.dump(self.model, '../checkpoint.pkl', compress=3)
        
    
    def load(self, path):
        joblib.load(path)
