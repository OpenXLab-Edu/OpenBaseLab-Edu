# from base import *
from LR import *
# from MMEdu import MMMlearing
dataset_path = "./test.csv" 
model = LR()
model.load_dataset(dataset_path) 
model.train()
acc = model.inference()
