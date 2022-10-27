
from BaseML import Classification
import numpy as np
from sklearn import datasets

# 导入sklearn内置的iris数据集进行测试
X = datasets.load_iris().data
y = datasets.load_iris().target

def iris_cls(algorithm = 'MLP'): # path指定模型保存的路径
    # 实例化模型
    model = Classification.cls(algorithm = algorithm)
    # 指定数据集格式
    model.load_dataset(X,y,type = 'numpy',x_column=[0,1,2,3], y_column=[0])
    # 开始训练
    model.train()
    # 构建测试数据
    test_data = [[0.2,0.4,3.2,5.6],
                [2.3,1.8,0.4,2.3]]
    test_data = np.asarray(test_data)
    result = model.inference(test_data)
    print(result)

    model.save()

if __name__ == '__main__':
    iris_cls(algorithm='Kmeans')