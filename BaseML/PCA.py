from sklearn.decomposition import PCA as pca_reduction
import os


class PCA:
    def __init__(self,
                 n_components='mle',
                 ):
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset = None
        # self.x_train, self.x_test = 0, 0
        self.model = pca_reduction(n_components=n_components)

    def train(self):
        self.model.fit(self.dataset)
        print(self.model.explained_variance_ratio_)
        # 返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。

    def inference(self, data):
        self.model.transform(data)
        print(self.model.n_features_)
        print(self.model.n_samples_)

    def load_dataset(self, dataset):
        self.dataset = dataset
