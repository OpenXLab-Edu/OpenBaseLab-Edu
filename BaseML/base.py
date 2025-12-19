# BaseML基类，各个大类能够继承其中的基本方法
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


class baseml:
    """BaseML中的继承基类,单例模式避免多次调用创建

    """

    def __init__(self):

        self.cwd = os.path.dirname(os.getcwd())  # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.x_train, self.x_test, self.y_train, self.y_test, self.x_val, self.y_val = [
        ], [], [], [], [], []
        self.X = []
        self.Y = []
        self.dataset = []
        self.model = None
        self.test_size = 0.2
        self.scaler = None
        self.demo_input = None
        self.input_shape = None

    # 采用单例，避免基类创建太多次
    def __new__(cls, *args, **kwargs):
        # print("__new__")
        if not hasattr(baseml, "_instance"):
            # print("创建新实例")
            baseml._instance = object.__new__(cls)
        return baseml._instance

    def train(self):
        # 必须要改写的类
        raise NotImplementedError("train function must be implemented")

    def inference(self):
        # 必须要改写的类
        raise NotImplementedError("inference function must be implemented")

    def load_dataset(self, X, y=[], type=None, x_column=[], y_column=[],
                     shuffle=True, show=False, split=True, scale=False):
        """Load the model's data set.

        Args:
            X (str|numpy|pandas|list): 自变量.
            y (str|numpy|pandas|list, optional): 目标值. 默认为 [].
            type (str, optional): X和y的输入格式, choice = ['csv', 'numpy','pandas','list','txt], 最后统一转换为numpy. 
            x_column (list, optional): X 的索引列. 默认设置为X的所有列.
            y_column (list, optional): y的索引列. 默认设置为y的所有列.
            shuffle (bool, optional): 是否对元素随机排序. 默认为True.
            show (bool, optional): 显示5条数据. 默认为True.
            split(bool, optional): 是否划分数据集为训练集和测试集. 默认为True.
            scale(bool, optional): 是否对数据进行归一化. False.

        """
        if (type == 'csv' or type == 'txt') and len(x_column) == 0:
            raise ValueError("请传入数据列号")
        if type == 'csv':
            self.dataset = pd.read_csv(X).values  # .values就转成numpy格式了
            if shuffle:
                np.random.shuffle(self.dataset)
            self.get_data(self.dataset, self.dataset,
                          x_column, y_column, split, scale)
        elif type == 'numpy':
            if shuffle:
                X, y = self.shuffle_data(X, y)
            self.get_data(X, y, x_column, y_column, split, scale)
        elif type == 'pandas':
            X = X.values
            y = y.values if len(y) > 0 else []
            if shuffle:
                X, y = self.shuffle_data(X, y)
            self.get_data(X, y, x_column, y_column, split, scale)
        elif type == 'list':
            X = np.array(X)
            y = np.array(y) if len(y) > 0 else []
            if shuffle:
                X, y = self.shuffle_data(X, y)
            self.get_data(X, y, x_column, y_column, split, scale)
        elif type == 'txt':
            self.dataset = np.loadtxt(X)
            self.dataset = self.dataset.values
            if shuffle:
                np.random.shuffle(self.dataset)
            self.get_data(self.dataset, self.dataset,
                          x_column, y_column, split, scale)

        print("Load dataset successfully!")

        if show and len(self.x_train) >= 5:   # 显示前5条数据
            print("X")
            print(self.x_train[:5])
            print("y")
            if len(self.y_train) >= 5:
                print(self.y_train[:5])
            else:
                print("None")

    def get_data(self, X, y, x_column, y_column, split, scale):
        """通过列号获取真实的训练数据

        Args:
            X (numpy.ndarray): 自变量.
            y (numpy.ndarray): 因变量.
            x_column (list): 自变量的列索引集合.
            y_column (list): 因变量的列索引集合.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(x_column) == 0 and len(X):
            # 如果没有赋值，那么默认选用所有列
            x_column = list(range(X.shape[1]))
        if len(y_column) == 0 and len(y):
            # 如果没有赋值，默认用y的所有列
            if y.ndim == 1:
                y_column = [0]
            else:
                y_column = list(range(y.shape[1]))

        if len(X):
            self.x_train = X[:, x_column]

        if scale:  # 对训练数据进行归一化，在聚类、部分分类的时候需要使用
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.x_train = self.scaler.fit_transform(self.x_train)

        if len(y):  #
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            self.y_train = y[:, y_column]
            if self.y_train.shape[0]:
                self.dataset = np.concatenate(
                    (self.x_train, self.y_train), axis=1)  # 按列进行拼接
                
        else:
            self.dataset = self.x_train

        if split:   # 进行数据集划分
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train,  self.y_train, test_size=self.test_size, random_state=42)

    def shuffle_data(self, X, y):
        if len(X) == len(y):
            c = list(zip(X, y))  # 保持X与y的对应关系
            np.random.shuffle(c)
            X = np.array([t[0] for t in c])
            y = np.array([t[1] for t in c])
        elif len(X) > 0 and len(y) == 0:
            np.random.shuffle(X)

        return X, y

    def save(self, path="checkpoint.pkl"):
        data = {
            'model': self.model,
            'input_shape': self.x_train.shape,
            'demo_input': self.x_train[:1],
        }
        print("Saving model checkpoints...")
        joblib.dump(data, path, compress=3)
        print("Saved successfully!")

    def load(self, path):
        # self.model = joblib.load(path)
        model = joblib.load(path)
        if isinstance(model, dict):
            self.model = model['model']
            try:
                self.demo_input = model['demo_input']
                self.input_shape = model['input_shape']
            except:
                pass
        else:
            self.model = model
        


    def reverse_scale(self, data):
        return self.scaler.inverse_transform(data)

    def get_test_data(self):
        return self.x_test, self.y_test

    def convert_np(self, data):
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, tuple):
            data = np.array(data)
        else:
            TypeError("The type {} is not supported".format(type(data)))
        return data

    def plot(self, X=None, y_true=None):
        # 模型可视化，若不被改写则不被支持
        raise NotImplementedError(
            "Error Code: -405. No implementation of this method.")

    def metricplot(self, X=None, y_true=None):
        # 模型可视化，若不被改写则不被支持
        raise NotImplementedError(
            "Error Code: -405. No implementation of this method.")

    def load_tab_data(self, data_path, train_val_ratio=1.0, shuffle=True,random_seed=42,y_type='float',**kw):
        # if y_type == 'long' and self.task_type == 'reg':
        #     y_type = 'float'
        data = np.loadtxt(data_path, dtype=float, delimiter=',',skiprows=1) # [120, 4]
        x = data[:,:-1]
        y = data[:, -1]
        y = y.astype(y_type)
        if 0 < train_val_ratio < 1:
            train_size =  int(train_val_ratio * len(x))
            val_size =  len(x) - train_size

            x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_size, test_size=val_size, random_state=random_seed,shuffle=shuffle)
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None

        # if self.task_type == 'cls':
        #     y_train = y_train.astype(int)
        #     y_val = y_val.astype(int) if y_val is not None else None
        # elif self.task_type =='reg':
        #     y_train = y_train.astype(float)
        #     y_val = y_val.astype(float) if y_val is not None else None
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_val
        self.y_test = y_val
        return x_train, y_train, x_val, y_val
    
    def set_para(self, **kw):
        for i in kw:
            print("Setting {} to {}".format(i, kw[i]))
            setattr(self.model, i, kw[i])

    @property
    def para(self):
        return self.para

    @para.setter
    def para(self, kw):
        for i in kw:
            print("Setting {} to {}".format(i, kw[i]))
            setattr(self.model, i, kw[i])

    def valid(self, path=None, x=None ,y=None ,metrics='accuracy'):
        """验证模型的准确率

        Args:
            path (str): 验证集的路径
            x (np.ndarray, optional): 验证集的特征. Defaults to None.
            y (np.ndarray, optional): 验证集的标签. Defaults to None.
            metrics (str, optional): 验证集的评估指标. Defaults to 'accuracy'.

        Returns:
            acc: 返回验证指标的值
            y_pred: 返回预测y值
        """
        if path is None and x is None and y is None: # 如果没有输入数据，默认采用x_test和y_test
            x = self.x_test
            y = self.y_test
        elif x is None and y is None: # 如果输入了路径，但是没有输入数据，则读取路径
            df = pd.read_csv(path)
            x = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            self.x_test = x
            self.y_test = y

        # 验证集的特征和标签不能为空
        assert x is not None and y is not None,  "Error Code: -801. The validation set cannot be empty. "
        assert len(x) > 0 and len(y) > 0,  "Error Code: -801. The validation set cannot be empty. "
        
        y_pred = self.inference(x)

        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,\
        r2_score,mean_squared_error,mean_absolute_error,auc,\
        silhouette_score
        if metrics == 'accuracy' or metrics=='acc':
            score = accuracy_score(y, y_pred)
            print('验证准确率为：{}%'.format(score * 100))
        elif metrics == 'precision':
            score = precision_score(y, y_pred,average='weighted')
            print('验证精确率为：{}%'.format(score * 100))

        elif metrics =='recall':
            score = recall_score(y, y_pred,average='weighted')
            print('验证召回率为：{}%'.format(score * 100))

        elif metrics == 'f1':
            score = f1_score(y, y_pred,average='weighted')
            print('验证f1-score为：{}%'.format(score * 100))
        
        elif metrics == 'auc':
            score = auc(y, y_pred)
            print('验证auc为：{}%'.format(score * 100))
        
        elif metrics == 'r2':
            assert len(y) >= 2, "Error Code: -603. The validation set has less than 2 samples and r2-score cannot be calculated."
            score = r2_score(y, y_pred)
            
            print('验证r2-score为：{}'.format(score))
            
        elif metrics =='mse':
            score = mean_squared_error(y, y_pred)
            print('验证均方误差为：{}'.format(score))

        elif metrics =='mae':
            score = mean_absolute_error(y, y_pred)
            print('验证平均绝对误差为：{}'.format(score))

        else:
            raise AssertionError("Error Code: -307. The '{}' metric is not currently supported.".format(metrics))
        return score,y_pred
