from sklearn.tree import DecisionTreeClassifier
import os


class CART:
    def __init__(self,
                 ):
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset = ''
        self.x_train, self.x_test = 0, 0
        self.model = DecisionTreeClassifier()

    def train(self):
        self.model.fit(self.dataset)
        print(self.model.explained_variance_ratio_)
        # 返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。

    def inference(self, data):
        self.model.fit_transform(data)
        print(self.model.n_features_)
        print(self.model.n_samples_)

    def load_dataset(self, dataset):
        self.dataset = dataset
