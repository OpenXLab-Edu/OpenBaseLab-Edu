import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib
import random

class cls:
    def __init__(self, algorithm='KNN', n_neighbors=5, n_estimators=100, N_CLUSTERS=5):
        self.algorithm = algorithm
        self.cwd = os.path.dirname(os.getcwd())  # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = ' '
        self.test_size = 0.2
        self.test_set = ' '
        self.x_train, self.x_test, self.y_train, self.y_test = 0, 0, 0, 0
        if self.algorithm == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif self.algorithm == 'SVM':
            self.model = SVC()
        elif self.algorithm == 'NaiveBayes':
            self.model = GaussianNB()
        elif self.algorithm == 'CART':
            self.model = DecisionTreeClassifier()
        elif self.algorithm == 'AdaBoost':
            self.model = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
        elif self.algorithm == 'Kmeans':
            self.n = N_CLUSTERS
            self.model = KMeans(self.n)
        elif self.algorithm == 'MLP':
            self.model = MLPClassifier(solver='lbfgs')

    def train(self, seed=0, data_type='csv'):
        if self.algorithm in ['AdaBoost','SVM','NaiveBayes', 'MLP']:
            np.random.seed(seed)
            np.random.shuffle(self.dataset)

            self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=0)
            self.model.fit(self.x_train, self.y_train)
        
        elif self.algorithm == 'CART':
            self.model.fit(self.dataset)
            print(self.model.explained_variance_ratio_)
            # 返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。
        
        elif self.algorithm == 'KNN':
            self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=0)
            self.model.fit(self.x_train, self.y_train)
            acc = self.model.score(self.x_test, self.y_test)
            print('准确率为：{}%'.format(acc * 100))
        
        elif self.algorithm == 'Kmeans':
            # 对列数据进行文本过滤，只抽取有数据的列
            delete_list = []
            if self.x_train.ndim >= 2:
                for col_idx in range(self.x_train.shape[1]):
                    col = self.x_train[:,col_idx]
                    # 随机取一个元素，查看其type
                    if isinstance(random.choice(col),str):
                        delete_list.append(col_idx)

            self.x_train = np.delete(self.x_train, delete_list, axis=1)
            self.model.fit(self.x_train)

    def inference(self, data = np.nan):
        if data is not np.nan: # 对data进行了指定
            self.x_test = data

        if self.algorithm in ['AdaBoost','SVM','NaiveBayes', 'MLP','KNN','CART']:
            pred = self.model.predict(self.x_test)
            return pred

        elif self.algorithm == 'Kmeans':
            labels = self.model.labels_      # 获取聚类标签
            print(silhouette_score(self.x_train, labels))      # 获取聚类结果总的轮廓系数
            print(self.model.cluster_centers_)	# 输出类簇中心
            for i in range(self.n):
                print(f" CLUSTER-{i+1} ".center(60, '='))
                print(self.dataset[labels == i])

            if data is not np.nan:
                pred = self.model.predict(data)
                return pred

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
        

    def save(self,path="checkpoint.pkl"):
        print("Saving model checkpoints...")
        joblib.dump(self.model, path, compress=3)
        
    
    def load(self, path):
        self.model = joblib.load(path)

    def get_data(self,X,y,x_column,y_column):
        if len(X):
            self.x_train = X[:,x_column]
        if len(y):  # 
            if y.ndim == 1:
                y = y.reshape(-1,1)
            self.y_train = y[:,y_column]
            if self.y_train.shape[0]:
                self.dataset = np.concatenate((self.x_train,self.y_train),axis=1) # 按列进行拼接
