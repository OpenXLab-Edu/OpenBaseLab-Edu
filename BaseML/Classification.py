import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class cls:
    def __init__(self, algorithm='KNN', n_neighbors=5, n_estimators=100, ):
        self.algorithm = algorithm
        self.cwd = os.path.dirname(os.getcwd())  # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = ' '
        self.test_size = ' '
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

    def train(self, seed=0, data_type='csv'):
        if self.algorithm == 'AdaBoost' or 'SVM' or 'NaiveBayes':
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
        
        elif self.algorithm == 'CART':
            self.model.fit(self.dataset)
            print(self.model.explained_variance_ratio_)
            # 返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。
        
        elif self.algorithm == 'KNN':
            self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.dataset['data'], self.dataset['target'], test_size=0.2, random_state=0)
            self.model.fit(self.x_train, self.y_train)
            acc = self.model.score(self.x_test, self.y_test)
            print('准确率为：{}%'.format(acc * 100))

    def inference(self, data):
        if self.algorithm == 'AdaBoost' or 'SVM' or 'NaiveBayes':
            pred = self.model.predict(self.test_set['data'])
            acc = accuracy_score(self.test_set['label'], pred)
            print('准确率为：{}%'.format(acc * 100))
        elif self.algorithm == 'KNN':
            result = self.model.predict(data)
            print(result)
            print("分类结果：{}".format(self.dataset['target_names'][result]))
        elif self.algorithm == 'CART':
            self.model.fit_transform(data)
            print(self.model.n_features_)
            print(self.model.n_samples_)

    def load_dataset(self, path, test_size=0.2, dataset=''):
        self.dataset_path = path
        self.test_size = test_size
        self.dataset=dataset
