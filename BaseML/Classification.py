import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib
import cv2
import random
from skimage.feature import hog
from scipy.spatial.distance import cdist

from demo.face import extract_hog_features

class cls:
    def __init__(self, algorithm='KNN', n_neighbors=5, n_estimators=100, N_CLUSTERS=5):
        self.algorithm = algorithm
        self.cwd = os.path.dirname(os.getcwd())  # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = ' '
        self.test_size = 0.2
        self.test_set = ' '
        self.x_train, self.x_test, self.y_train, self.y_test = 0, 0, 0, 0
        self.X = 0
        self.Y = 0
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
        elif self.algorithm == 'RandomForest':
            self.model == RandomForestClassifier(n_estimators=n_estimators,random_state=0)

    def train(self, seed=0, data_type='csv'):
        if self.algorithm in ['AdaBoost','SVM','NaiveBayes', 'MLP']:
            np.random.seed(seed)
            np.random.shuffle(self.dataset)

            self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=0)
            self.model.fit(self.x_train, self.y_train)
        
        elif self.algorithm == 'CART':
            self.model.fit(self.x_train, self.y_train)
            # print(self.model.explained_variance_ratio_)
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
        
        elif self.algorithm == 'RandomForest':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=42)
            self.model.fit(self.x_train, self.y_train)


    def inference(self, data = np.nan):
        if data is not np.nan: # 对data进行了指定
            self.x_test = data

        if self.algorithm in ['AdaBoost','SVM','NaiveBayes', 'MLP','KNN','CART', 'RandomForest']:
            self.pred = self.model.predict(self.x_test)
            return self.pred

        elif self.algorithm == 'Kmeans':
            labels = self.model.labels_      # 获取聚类标签
            print(silhouette_score(self.x_train, labels))      # 获取聚类结果总的轮廓系数
            print(self.model.cluster_centers_)	# 输出类簇中心
            for i in range(self.n):
                print(f" CLUSTER-{i+1} ".center(60, '='))
                print(self.dataset[labels == i])

            pred = self.model.predict(self.x_test)
            return pred


    # 从文件加载数据集，支持csv文件和txt文件
    def load_dataset_from_file(self, path, x_column = [], y_column = []):
        if type == 'csv':
            self.dataset = pd.read_csv(path).values # .values就转成numpy格式了
            X = self.dataset[:,x_column]
            y = self.dataset[:,y_column]
            self.get_data(X,y,x_column,y_column)
        elif type == 'txt':
            self.dataset = np.loadtxt(path)
            X = self.dataset[:,x_column]
            y = self.dataset[:,y_column]
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
            self.get_data(self.dataset,self.dataset,x_column,y_column)
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
            self.dataset = self.dataset.values
            self.get_data(self.dataset,self.dataset,x_column,y_column)
        

    def save(self,path="checkpoint.pkl"):
        print("Saving model checkpoints...")
        joblib.dump(self.model, path, compress=3)
        print("Saved successfully!")
        
    
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

    def extract_hog_features(self, X):
        image_descriptors = []
        for i in range(len(X)):                                         # 此处的X为之前训练部分所有图像的矩阵形式拼接而来，
            # print(i)                                                  # 所以len(X)实为X中矩阵的个数，即训练部分图像的个数
            fd, _ = hog(X[i], orientations=9, pixels_per_cell=(16, 16), cells_per_block=(16, 16),
                                block_norm='L2-Hys', visualize=True)    # 此处的参数细节详见其他文章
            image_descriptors.append(fd)                                # 拼接得到所有图像的hog特征
        return image_descriptors                                        # 返回的是训练部分所有图像的hog特征


    def extract_hog_features_single(self, X):
        image_descriptors_single = []
        fd, _ = hog(X, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(16, 16),
                                block_norm='L2-Hys', visualize=True)
        image_descriptors_single.append(fd)
        return image_descriptors_single

    
    def read_data_and_pre(self, label2id, path):
        X = []
        Y = []
        path =r'../dataset/JAFFE/training_set'
        for label in os.listdir(path):
            for img_file in os.listdir(os.path.join(path, label)):              # 遍历
                image = cv2.imread(os.path.join(path, label, img_file))         # 读取图像
                result = image/255.0                                            # 图像归一化
                res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                   # 转灰度
                # cv2.waitKey(0)
                cv2.destroyAllWindows()
                X.append(res)                                                   # 将读取到的所有图像的矩阵形式拼接在一起
                Y.append(label2id[label])                                       # 将读取到的所有图像的标签拼接在一起
        self.X = self.extract_hog_features(X)
        self.Y = Y
        return self.extract_hog_features(X), Y                                                             # 返回的X,Y分别是图像的矩阵表达和图像的标签


    def plot(self):
        acc = accuracy_score(self.y_test, self.pred)
        precision = precision_score(self.y_test, self.pred, average='macro')
        recall = recall_score(self.y_test, self.pred, average='macro')
        cm = confusion_matrix(self.y_test, self.pred)
        print(cm)
        print('Acc: ', acc)
        print('Precision: ', precision)
        print('Recall: ', recall)
        
        xtick = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        ytick = xtick
        
        f, ax = plt.subplots(figsize=(7, 5))
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=15)
        
        seaborn.set(font_scale=1.2)
        plt.rc('font',family='Times New Roman', size=15)
        
        seaborn.heatmap(cm,fmt='g', cmap='Blues', annot=True, cbar=True,xticklabels=xtick, yticklabels=ytick, ax=ax)
        
        plt.title('Confusion Matrix', fontsize='x-large')
        
        plt.show()

    
    def test(self, path):
    # 下面为同一文件夹下多张图片的表情识别
    # labelid2 = {0:'angry',1: 'disgust',2: 'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
        path=path
        i = 1
        for dir in os.listdir((path)):
            temp_path = os.path.join(path, dir)
            for image_file in os.listdir(temp_path):
                image = cv2.imread(os.path.join(temp_path,image_file))
                # result = image/255.0
                # cv2.waitKey(1000)
                # cv2.destroyAllWindows()
                result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                X_Single = self.extract_hog_features_single(result)
                predict = self.model.predict(X_Single)         # 可以在这里选择分类器的类别
                print(i)
                i += 1
                if predict == 0:
                    print('angry')
                elif predict == 1:
                    print('disgust')
                elif predict == 2:
                    print('fear')
                elif predict == 3:
                    print('happy')
                elif predict == 4:
                    print('neutral')
                elif predict == 5:
                    print('sad')
                elif predict == 6:
                    print('surprise')
