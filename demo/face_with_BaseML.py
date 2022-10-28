import os
from re import X
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
import seaborn
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from BaseML import Classification, Regression

def Face():
    label2id = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    data_path =r'../dataset/JAFFE/training_set'
    test_path =r'../dataset/JAFFE/testing_set'

    Forest = Classification.cls('RandomForest', n_estimators=180)
    Forest.read_data_and_pre(label2id, path=data_path)
    Forest.train()
    Forest.save()
    Forest.inference()
    Forest.plot()
    Forest.test()

if __name__ == '__main__':
    Face()