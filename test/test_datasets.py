from BaseML import Classification
import numpy as np

DATA_PATH = 'Downloads/China_cities.csv'              # 数据集路径

if __name__ == '__main__':

    model = Classification.cls(algorithm = 'Kmeans', N_CLUSTERS=5)
    # 指定数据集的路径
    model.load_dataset_from_file(path="Downloads/China_cities.csv",x_column=[0,1], y_column=[0])

    # 开始训练
    model.train()
    # kmeans输出聚类结果，不需要输入数据
    model.inference()
    # 模型保留
    model.save()