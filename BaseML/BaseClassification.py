
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from yellowbrick.classifier import ClassPredictionError
from .base import baseml


class Classification(baseml):
    """BaseML中的分类模块,包含['KNN'(K近临分类), 'SVM'(支持向量机分类), 'NaiveBayes'(朴素贝叶斯分类), 'CART'(决策树分类), 
        'AdaBoost'(自适应增强分类), 'MLP'(多层感知机分类), 'RandomForest'(随机森林分类)]分类算法.

    Attributes:
        algorithm: 算法名称
        model: 实例化的模型

    更多用法及算法详解可参考：https://xedu.readthedocs.io/zh/master/baseml/introduction.html
    """

    def __init__(self, algorithm='KNN', n_neighbors=5, n_estimators=100, n_hidden=(100,), para={}):
        """cls类初始化.

        Args:
            algorithm (str, optional): 采用的分类算法. Defaults to 'KNN'.
            n_neighbors (int, optional): KNN的k值. Defaults to 5.
            n_estimators (int, optional): Adaboost|RandomForest所集成的决策树个数. Defaults to 100.
            n_hidden (tuple, optional): MLP隐藏层的形状. Defaults to (100,).
            para (dict, optional): 参数字典,可自定义参数放入,参数名称详见sklearn官方文档. Defaults to {}.
        """
        super(Classification, self).__init__()   # 继承父类的构造方法
        self.algorithm = algorithm

        if self.algorithm == 'KNN':
            if len(para) > 1:
                self.model = KNeighborsClassifier(**para)
            else:
                self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif self.algorithm == 'SVM':
            if len(para) > 1:
                self.model = SVC(**para)
            else:
                self.model = SVC(probability=True)
        elif self.algorithm == 'NaiveBayes':
            if len(para) > 1:
                self.model = GaussianNB(**para)
            else:
                self.model = GaussianNB()
        elif self.algorithm == 'CART':
            if len(para) > 1:
                self.model = DecisionTreeClassifier(**para)
            else:
                self.model = DecisionTreeClassifier()
        elif self.algorithm == 'AdaBoost':
            if len(para) > 1:
                self.model = AdaBoostClassifier(**para)
            else:
                self.model = AdaBoostClassifier(
                    n_estimators=n_estimators, random_state=0)

        elif self.algorithm == 'MLP':
            if len(para) > 1:
                self.model = MLPClassifier(**para)
            else:
                self.model = MLPClassifier(
                    hidden_layer_sizes=n_hidden, solver='lbfgs')
        elif self.algorithm == 'RandomForest':
            if len(para) > 1:
                self.model = RandomForestClassifier(**para)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=0)

    def train(self, validate=False,val_size=0.2, lr=0.001,epochs=200):
        """训练模型.

        Args:
            validate (bool, optional): 是否需要验证模型，并输出准确率. Defaults to False.
            val_size (float, optional): 验证集比例. Defaults to 0.2.
            lr (float, optional): 学习率. Defaults to 0.001.
            epochs (int, optional): 训练轮数. Defaults to 200.
        """
        if self.algorithm in ['AdaBoost', 'SVM', 'NaiveBayes', 'MLP', 'KNN', 'CART', 'RandomForest']:
            # 设定学习率
            if self.algorithm == 'MLP':
                self.model.learning_rate_init = lr
                self.model.max_iter = epochs
            elif self.algorithm == 'AdaBoost':
                self.model.learning_rate = lr


            if validate:

                self.x_train, self.x_val, self.y_train, self.y_val = \
                    train_test_split(self.x_train, self.y_train,
                                     test_size=val_size, random_state=0)

            self.model.fit(self.x_train, self.y_train)

            if validate:
                pred = self.model.predict(self.x_val)
                acc = accuracy_score(self.y_val, pred)
                print('训练准确率为：{}%'.format(acc * 100))

    def inference(self, data=np.nan, verbose=True):
        """使用模型进行推理

        Args:
            data (np.ndarray, optional): 放进来推理的数据,不填默认使用self.x_test.
            verbose (bool, optional): 是否输出推理中的中间结果. Defaults to True.

        Returns:
            pred: 返回预测结果.
        """
        if data is not np.nan:  # 对data进行了指定
            x_test = self.convert_np(data)
            if self.input_shape is not None: 
                model_input_shape = str(self.input_shape).replace(str(self.input_shape[0]), 'batch')
                assert type(self.demo_input) == type(x_test), f"Error Code: -309. The data type {type(x_test)} doesn't match the model input type {type(self.demo_input)}. Example input: {self.demo_input.tolist()}."
                assert self.input_shape[1:] == x_test.shape[1:], f"Error Code: -309. The data shape {x_test.shape} doesn't match the model input shape {model_input_shape}. Example input: {self.demo_input.tolist()}."


        elif len(self.x_train) > 0 and len(self.x_test) == 0:
            x_test = self.x_train
        else:
            x_test = self.x_test
        x_test = self.convert_np(x_test)

        if self.algorithm in ['AdaBoost', 'SVM', 'NaiveBayes', 'MLP', 'KNN', 'CART', 'RandomForest']:
            pred = self.model.predict(x_test).astype(int)
            return pred

    def metricplot(self, X=None, y_true=None):
        """绘制模型分类准确率图, 可直观查看每一类的分类正误情况

        Args:
            X (np.ndarray, optional): 放入的测试数据, 不填默认使用self.x_test.
            y_true (np.ndarray, optional): 放入的测试数据的真实标签, 不填默认使用self.y_test.
        """

        assert len(self.x_train) > 0 and len(self.y_train) > 0,  \
            "Error Code: -601. No dataset is loaded."
        if X is None and y_true is None:
            assert len(self.x_test) > 0 and len(
                self.y_test) > 0,  "Error Code: -602. Dataset split was not performed."
            X = self.x_test
            y_true = self.y_test
        assert len(X) > 0 and len(y_true) > 0
        visualizer = ClassPredictionError(
            self.model
        )
        visualizer.fit(self.x_train, self.y_train)
        visualizer.score(X, y_true.reshape(-1))
        visualizer.show()

    def plot(self, X=None, y_true=None):
        """绘制分类模型图

        Args:
            X (np.ndarray, optional): 放入的测试数据, 不填默认使用self.x_test.
            y_true (np.ndarray, optional): 放入的测试数据的真实标签, 不填默认使用self.y_test.
        """

        # 如果没有任何输入，默认采用x_test和y_test
        if X is None:
            assert len(
                self.x_test) > 0, "Error Code: -602. Dataset split was not performed."
            X = self.x_test
        X = self.convert_np(X)
        y_pred = self.inference(X)
        if y_true is not None:
            y_true = self.convert_np(y_true)
        X = X.reshape(X.shape[0], -1)   # 转为二维
        if self.algorithm == 'KNN':
            self.knn_plot(X, y_pred, y_true)
        elif self.algorithm == 'SVM':
            self.svm_plot(X, y_pred, y_true)
        else:
            raise AssertionError(
                "Error Code: -405. No implementation of this method.")

    def knn_plot(self, X, y_pred, y_true=None):
        """绘制KNN分类图, 不同标签的样本用不同颜色点代替。选择2维特征作为xy坐标, 最多选择5个类别进行可视化。

        Args:
            X (np.ndarray): 放入的测试数据。
            y_pred (np.ndarray): 放入的测试数据的预测标签。
            y_true (np.ndarray, optional): 放入的测试数据的真实标签。
        """

        # 训练数据特征多于2维，仅取前两维
        if X.shape[1] > 2:
            print('\033[1;34;1mFeatures is more than 2 dimensions, '
                  'the first two dimensions are used by default.\033[0m')

        label = np.unique(y_pred)
        # 最多选择5个类别进行可视化
        if len(label) > 5:
            label.sort()
            label = label[:5]
            y_max = label[4]
            idx = np.where(y_pred <= y_max)
            y_pred = y_pred[idx]
            X = X[idx, :].squeeze()
            print('\033[1;34;1mThe number of classes is more than 5, '
                  'the top 5 classes are used by default.\033[0m')

        label_list = ["y_pred_" + str(i) for i in range(len(label))]
        y_pred_plot = plt.scatter(
            X[:, 0], X[:, 1], marker='o', c=y_pred, cmap='rainbow')
        handles = y_pred_plot.legend_elements()[0]

        # 只有显式输入y_true才会被画出
        if y_true is not None:
            true_label = np.unique(y_true)
            true_label_list = ["y_true_" + str(i)
                               for i in range(len(true_label))]
            y_true_plot = plt.scatter(
                X[:, 0], X[:, 1], marker='s', c=y_true, cmap='viridis', s=10)
            handles += y_true_plot.legend_elements()[0]
            label_list += true_label_list

        plt.legend(handles=handles, labels=label_list)
        plt.show()

    def svm_plot(self, X, y_pred, y_true=None):
        """绘制SVM分类图, 不同标签的样本用不同颜色点代替, 绘制出SVM分类边界。选择2维特征作为xy坐标。

        Args:
            X (np.ndarray): 放入的测试数据。
            y_pred (np.ndarray): 放入的测试数据的预测标签。
            y_true (np.ndarray, optional): 放入的测试数据的真实标签。
        """

        assert self.model.n_features_in_ == 2, "Error Code: -306. "\
            "The number of features for training is wrong, required {}, "\
            "which is {}.".format(2, self.model.n_features_in_)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap="rainbow")
        if y_true is not None:
            ax.scatter(X[:, 0], X[:, 1], c=y_true,
                       s=8, cmap="viridis", marker='s')
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = np.linspace(xlim[0], xlim[1], 30)  # 产生30个间隔
        y = np.linspace(ylim[0], ylim[1], 30)  # 产生30个间隔
        _Y, _X = np.meshgrid(y, x)
        z = self.model.predict(np.c_[_X.flatten(), _Y.flatten()])

        zz = z.reshape(_X.shape)

        ax.contour(_X, _Y, zz, colors="k",
                   levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.show()

    