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

   # 从文件加载数据集，支持csv文件和txt文件
    def load_dataset_from_file(self, path, x_column = [], y_column = []):
        if type == 'csv':
            self.dataset = pd.read_csv(path).values # .values就转成numpy格式了
            self.get_data(X,y,x_column,y_column)
        elif type == 'txt':
            self.dataset = np.loadtxt(path)
            X = X.values
            y = y.values
            self.get_data(X,y,x_column,y_column)

    # 从数据加载数据集，支持['numpy','list','DataFrame']
    def load_dataset_from_data(self, X, y = None, x_column = [], y_column = []):
        if type(X) != type(y):
            raise TypeError("数据格式不同，无法加载")
        if isinstance(X,list):
            X = np.array(X)
            y = np.array(y)
            self.get_data(X,y,x_column,y_column)
        elif isinstance(X,np.ndarray):
            self.get_data(X,y,x_column,y_column)
        elif isinstance(X,pd.DataFrame):
            X = X.values
            y = y.values
            self.get_data(X,y,x_column,y_column)



    # 支持的type有['csv', 'numpy','pandas','list','txt]，后面一律转为numpy格式
    def load_dataset(self, X, y = None, type = None, x_column = [], y_column = []):
        if len(x_column) == 0:
            raise ValueError("请传入数据列号")
        if type == 'csv':
            self.dataset = pd.read_csv(X).values # .values就转成numpy格式了
            self.get_data(X,y,x_column,y_column)
        elif type == 'numpy':  # 统一转成numpy格式
            self.get_data(X,y,x_column,y_column)
        elif type == 'pandas':
            X = X.values
            y = y.values
            self.get_data(X,y,x_column,y_column)
        elif type == 'list':
            X = np.array(X)
            y = np.array(y)
            self.get_data(X,y,x_column,y_column)
        elif type == 'txt':
            self.dataset = np.loadtxt(X)
            X = X.values
            y = y.values
            self.get_data(X,y,x_column,y_column)
        
    def get_data(self,X,y,x_column,y_column):
        if len(X):
            self.x_train = X[:,x_column]
        if len(y):  # 
            if y.ndim == 1:
                y = y.reshape(-1,1)
            self.y_train = y[:,y_column]
            if self.y_train.shape[0]:
                self.dataset = np.concatenate((self.x_train,self.y_train),axis=1) # 按列进行拼接


    def save(self):
        print("Saving model checkpoints...")
        joblib.dump(self.model, '../checkpoint.pkl', compress=3)
        
    
    def load(self, path):
        joblib.load(path)