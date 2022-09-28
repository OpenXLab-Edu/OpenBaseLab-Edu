from BaseML import Classification

N_CLUSTERS = 7                                     # 类簇的数量
DATA_PATH = '~/Downloads/China_cities.csv'              # 数据集路径


def city():
    model = Classification(algorithm='Kmeans', N_CLUSTERS=5)
    model.load_dataset(path=DATA_PATH)
    model.train()
    model.inference()
    model.save()
