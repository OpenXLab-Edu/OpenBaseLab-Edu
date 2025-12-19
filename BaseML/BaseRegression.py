
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from yellowbrick.regressor import PredictionError
import matplotlib.pyplot as plt
import joblib
from .base import baseml


class Regression(baseml):
    """BaseML中的回归模块,包含['LinearRegression'(线性回归), 'CART'(决策树回归), 'RandomForest'(随机森林回归),
       'Polynomial'(多项式回归), 'Lasso'(角回归), 'Ridge'(岭回归), 'SVM'(支持向量机回归), 'AdaBoost'(自适应增强回归), 'MLP'(多层感知机回归)]回归算法.

    Attributes:
        algorithm: 算法名称
        model: 实例化的模型
    
    更多用法及算法详解可参考：https://xedu.readthedocs.io/zh/master/baseml/introduction.html
    """

    def __init__(self, algorithm='LinearRegression', n_estimators=20, degree=2, n_hidden=(100,), para={}):
        """reg类的初始化

        Args:
            algorithm (str, optional): 选择的回归学习器. Defaults to 'LinearRegression'.
            n_estimators (int, optional): RandomForest集成的决策树个数. Defaults to 20.
            degree (int, optional): 多项式回归的阶数. Defaults to 2.
            para (dict, optional): 参数字典,可自定义参数放入,参数名称详见sklearn官方文档. Defaults to {}.
        """
        super(Regression, self).__init__()   # 继承父类的构造方法
        self.algorithm = algorithm
        if self.algorithm == 'LinearRegression':   # 线性回归
            if len(para) > 1:
                self.model = linear_model.LinearRegression(**para)
            else:
                self.model = linear_model.LinearRegression()
        elif self.algorithm == 'CART':   # 决策树回归
            if len(para) > 1:
                self.model = tree.DecisionTreeRegressor(**para)
            else:
                self.model = tree.DecisionTreeRegressor()
        elif self.algorithm == 'RandomForest':   # 随机森林回归
            if len(para) > 1:
                self.model = ensemble.RandomForestRegressor(**para)
            else:
                self.model = ensemble.RandomForestRegressor(
                    n_estimators=n_estimators)
        elif self.algorithm == 'Polynomial':     # 多项式回归
            if len(para) > 1:
                self.model = PolynomialFeatures(**para)
                self.poly_linear_model = linear_model.LinearRegression()
            else:
                self.model = PolynomialFeatures(degree=degree)
                self.poly_linear_model = linear_model.LinearRegression()
        elif self.algorithm == 'Lasso':          # Lasso回归
            if len(para) > 1:
                self.model = linear_model.Lasso(**para)
            else:
                self.model = linear_model.Lasso()
        elif self.algorithm == 'Ridge':          # 岭回归
            if len(para) > 1:
                self.model = linear_model.Ridge(**para)
            else:
                self.model = linear_model.Ridge()
        elif self.algorithm == 'SVM':
            if len(para) > 1:
                self.model = SVR(**para)
            else:
                self.model = SVR(degree=degree)
        elif self.algorithm == 'AdaBoost':
            if len(para) > 1:
                self.model = AdaBoostRegressor(**para)
            else:
                self.model = AdaBoostRegressor(n_estimators=n_estimators)
        elif self.algorithm == 'MLP':
            if len(para) > 1:
                self.model = MLPRegressor(**para)
            else:
                self.model = MLPRegressor(
                    hidden_layer_sizes=n_hidden, solver='lbfgs')

    def train(self, validate=False,val_size=0.2, lr=0.001,epochs=200):
        """训练模型.

        Args:
            validate (bool, optional): 是否需要验证模型，并输出准确率. Defaults to False.
            val_size (float, optional): 验证集比例. Defaults to 0.2.
            lr (float, optional): 学习率. Defaults to 0.001.
            epochs (int, optional): 训练轮数. Defaults to 200.
        """
        if self.algorithm == 'MLP':
            self.model.learning_rate_init = lr
            self.model.max_iter = epochs
        elif self.algorithm == 'AdaBoost':
            self.model.learning_rate = lr

        if validate:  # 需要划分数据集，并输出准确率
            self.x_train, self.x_val, self.y_train, self.y_val = \
                train_test_split(self.x_train, self.y_train,
                                 test_size=val_size, random_state=0)

        if self.algorithm == 'Polynomial':
            x_transformed = self.model.fit_transform(
                self.x_train)  # x每个数据对应的多项式系数
            self.poly_linear_model.fit(x_transformed, self.y_train)

        else:
            self.model.fit(self.x_train, self.y_train)

        if self.algorithm == 'LinearRegression':
            self.coef = self.model.coef_
            self.intercept = self.model.intercept_

        if validate:
            if len(self.y_val < 2):
                print("测试集小于2个样本，无法使用R值计算")
            else:
                pred = self.model.predict(self.x_val)
                acc = r2_score(self.y_val, pred)
                print('R值为: {}%'.format(acc))

    def inference(self, data=np.nan):
        """_summary_

        Args:
            data (numpy, optional): 放进来推理的数据,不填默认使用self.x_test.

        Returns:
            pred: 返回预测结果.
        """
        # if data is not np.nan:  # 对data进行了指定
        #     self.x_test = data
        x_test = data if data is not np.nan else self.x_test
        assert len(x_test) > 0, "Error Code: -601. No dataset is loaded."
        x_test = self.convert_np(x_test)
        if self.input_shape is not None: 
            model_input_shape = str(self.input_shape).replace(str(self.input_shape[0]), 'batch')
            assert type(self.demo_input) == type(x_test), f"Error Code: -309. The data type {type(x_test)} doesn't match the model input type {type(self.demo_input)}. Example input: {self.demo_input.tolist()}."
            assert self.input_shape[1:] == x_test.shape[1:], f"Error Code: -309. The data shape {x_test.shape} doesn't match the model input shape {model_input_shape}. Example input: {self.demo_input.tolist()}."

        if x_test.ndim != 2:
            x_test = x_test.reshape(x_test.shape[0], -1)
        if self.algorithm == 'Polynomial':
            x_trans = self.model.transform(x_test)
            self.pred = self.poly_linear_model.predict(x_trans)
            # self.pred = self.model.
        else:
            self.pred = self.model.predict(x_test)

        return self.pred

    # 重写方法
    def save(self, path="checkpoint.pkl"):
        print("Saving model checkpoints...")

        if self.algorithm == 'Polynomial':
            modelList = [self.model, self.poly_linear_model]
            data = {
                'model': modelList,
                'input_shape': self.x_train.shape,
                'demo_input': self.x_train[:1],
            }
            joblib.dump(data, path, compress=3)
        else:
            data = {
                'model': self.model,
                'input_shape': self.x_train.shape,
                'demo_input': self.x_train[:1],
            }
            joblib.dump(data, path, compress=3)
        print("Saved successfully!")

    def load(self, path):
        if self.algorithm == 'Polynomial':
            
            self.model = joblib.load(path)['model'][0]
            self.poly_linear_model = joblib.load(path)['model'][1]
        else:
            self.model = joblib.load(path)['model']

    def metricplot(self, X=None, y_true=None):
        """绘制模型回归预测误差图, 图中的identity为基准线, 说明预测出的标签(y轴)与
        真实标签(x轴)相同。回归模型越靠近基准线则越好。该图显示了回归模型的方差大小。

        Args:
            X (np.ndarray, optional): 放入的测试数据, 不填默认使用self.x_test.
            y_true (np.ndarray, optional): 放入的测试数据的真实标签, 不填默认使用self.y_test.
        """

        if X is None and y_true is None:
            X = self.x_test
            y_true = self.y_test
        # assert len(self.x_train) > 0 and len(self.y_train) > 0,  \
        #     "Error Code: -601. No dataset is loaded."
        assert X is not None and y_true is not None,  "Error Code: -604. No valid data is provided or the validataion dataset is empty."
        assert len(X) > 0 and len(y_true) > 0,  "Error Code: -604. No valid data is provided or the validataion dataset is empty."
        if self.algorithm == 'Polynomial':
            from sklearn.pipeline import make_pipeline
            model = make_pipeline(self.model, self.poly_linear_model)
        else:
            model = self.model

        visualizer = PredictionError(
            model,
            title="Actual vs. Predicted Values",
        )

        # self.y_test = self.y_test.squeeze()
        # visualizer.fit(self.x_train, self.y_train)
        # visualizer.score_ = visualizer.estimator.score(
        #     self.x_test, self.y_test)
        # result = self.inference(self.x_test).squeeze()
        # visualizer.draw(self.y_test, result)
        y_true = y_true.squeeze()
        visualizer.fit(self.x_train, self.y_train)  # 1. Fit on TRAIN
        visualizer.score(X, y_true)  # 2. Score on TEST (X, y_true)

        visualizer.show()

    def plot(self, X=None, y_true=None):
        """绘制回归模型图.

        Args:
            X (np.ndarray, optional): 放入的测试数据, 不填默认使用self.x_test.
            y_true (np.ndarray, optional): 放入的测试数据的真实标签, 不填默认使用self.y_test.
        """

        # 如果没有任何输入，默认采用x_test和y_test
        if X is None:
            assert len(
                self.x_test) is not None, "Error Code: -601. No dataset is loaded."
            X = self.x_test
            y_true = self.y_test
        X = self.convert_np(X)
        y_pred = self.inference(X)
        if y_true is not None:
            y_true = self.convert_np(y_true)
        X = X.reshape(X.shape[0], -1)   # 转为二维
        if self.algorithm == 'LinearRegression':
            self.linear_reg_plot(X[:, 0], y_pred, y_true)
        else:
            raise AssertionError(
                "Error Code: -405. No implementation of this method.")

    def linear_reg_plot(self, X, y_pred, y_true=None):
        """绘制线性回归模型图, 仅支持使用1维特征训练的模型. 

        Args:
            X (np.ndarray): 放入的测试数据.
            x_pred (np.ndarray): 测试数据的预测标签.
            y_true (np.ndarray, optional): 放入的测试数据的真实标签, 当被显式填入时才会绘制出真实的散点.
        """

        assert self.model.n_features_in_ == 1, \
            "Error Code: -306. "\
            "The number of features for training is wrong, required {}, "\
            "which is {}.".format(1, self.model.n_features_in_)
        fig, ax = plt.subplots()
        if y_true is not None:
            ax.scatter(X, y_true)
        ax.plot(X, y_pred, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('tight')

        plt.show()
