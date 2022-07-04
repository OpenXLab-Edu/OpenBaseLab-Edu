# from base import *
from Classifer import *
# from MMEdu import MMMlearing
dataset_path = "./test.csv" 
model = Classifer(backbone ='SVM')
model.load_dataset(dataset_path) 
model.train()
acc = model.inference()
