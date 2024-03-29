from turtle import back
import pandas as pd 
import numpy as np
import os 
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn import linear_model

class LR:
    def __init__(self,):
        self.cwd = os.path.dirname(os.getcwd())  #获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.model = linear_model.LinearRegression()
        self.dataset_path = ' '
        self.test_size = ' '
       
    def train(self, seed=0, data_type='csv'):
        np.random.seed(seed)
        if data_type == 'csv':
            dataset = pd.read_csv(self.dataset_path,sep=',',header=None).values
        np.random.shuffle(dataset)

        data, label = dataset[:,:-1],dataset[:,-1]
        train_index = int((1-self.test_size)*len(dataset))
        train_data, train_label = data[:train_index,],label[:train_index]
        self.test_set = {
            'data': data[train_index:,],
            'label': label[train_index:]
        }
        self.model.fit(train_data,train_label)

    def inference(self, mode='cls'):
        pred = self.model.predict(self.test_set['data']) 
        loss = mean_squared_error(self.test_set['label'],pred)
        print('Loss: {}'.format(loss))

    def load_dataset(self,path,test_size=0.2):
        self.dataset_path = path 
        self.test_size = test_size
