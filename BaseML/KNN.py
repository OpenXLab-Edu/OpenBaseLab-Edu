from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os


class KNN:
    def __init__(self,
                 n_neighbors=10,
                 ):
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.dataset = ''
        self.x_train, self.x_test, self.y_train, self.y_test = 0, 0, 0, 0
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        acc = self.model.score(self.x_test, self.y_test)
        print('准确率为：{}%'.format(acc * 100))

    def inference(self, data):
        result = self.model.predict(data)
        print(result)
        print("分类结果：{}".format(self.dataset['target_names'][result]))

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.dataset['data'], self.dataset['target'], test_size=0.2, random_state=0)
