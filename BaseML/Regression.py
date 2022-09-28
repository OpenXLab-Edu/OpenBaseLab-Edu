import os
from turtle import back
import pandas as pd 
import numpy as np
from sklearn.linear_model import Perceptron as per
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.decomposition import PCA as pca_reduction
import joblib

class reg:
    def __init__(self, algorithm='', n_components='mle'):
        self.cwd = os.path.dirname(os.getcwd())  #获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.algorithm = algorithm
        self.dataset_path = ' '
        self.test_size = ' '
        if self.algorithm ==  'LR':
            self.model = linear_model.LinearRegression()
        elif self.algorithm ==  'Perceptron':
            self.model = per()
        elif self.algorithm ==  'PCA':
            self.model = pca_reduction(n_components=n_components)
       
    def train(self, seed=0, data_type='csv'):
        if self.algorithm ==  'LR':
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
        elif self.algorithm ==  'Perceptron' or 'PCA':
            self.model.fit(self.dataset)
            print(self.model.explained_variance_ratio_)
            # 返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。

    def inference(self, data):
        if self.algorithm ==  'LR':
            pred = self.model.predict(self.test_set['data']) 
            loss = mean_squared_error(self.test_set['label'],pred)
            print('Loss: {}'.format(loss))
        elif self.algorithm ==  'Perceptron' or 'PCA':
            self.model.fit_transform(data)
            print(self.model.n_features_)
            print(self.model.n_samples_)

    def load_dataset(self,path,test_size=0.2, dataset=''):
        self.dataset_path = path 
        self.test_size = test_size
        self.dataset = dataset


    def save(self):
        print("Saving model checkpoints...")
        joblib.dump(self.model, '../checkpoint.pkl', compress=3)
        
    
    def load(self, path):
        joblib.load(path)