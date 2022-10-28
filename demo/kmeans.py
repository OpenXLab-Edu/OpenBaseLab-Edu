from BaseML import Classification
import numpy as np

N_CLUSTERS = 7                                     # 类簇的数量
DATA_PATH = 'dataset/China_cities.csv'              # 数据集路径


def city():
    # 实例化模型
    model = Classification.cls(algorithm = 'Kmeans', N_CLUSTERS=5)
    # 指定数据集的路径
    model.load_dataset(path=DATA_PATH)
    # 开始训练
    model.train()
    # kmeans输出聚类结果，不需要输入数据
    model.inference()
    # 模型保留
    model.save()


def kmeans_train(num_cluster,model_path):
    # 实例化模型
    model = Classification.cls(algorithm = 'Kmeans', N_CLUSTERS=num_cluster)
    # 指定数据集的路径
    model.load_dataset(DATA_PATH, type='csv', x_column=[2,3], y_column=[0])
    # 开始训练
    model.train()
    # 模型保存
    model.save(model_path)

def kmeans_inference(num_cluster,model_path):
    # 实例化模型
    model = Classification.cls(algorithm = 'Kmeans', N_CLUSTERS=num_cluster)
    # 加载模型数据集
    model.load_dataset(DATA_PATH, type='csv', x_column=[2,3], y_column=[0])
    # 加载模型权重文件
    model.load(model_path)
    # 进行推理
    model.inference()


if __name__ == '__main__':
    # city()
    kmeans_train(5,'checkpoint.pkl')
    kmeans_inference(5,'checkpoint.pkl')