# !/usr/bin/python3

import keras
from keras import layers, Sequential
import numpy as np
import pandas as pd
import os
import tensorflow as tf

import LinearRegression as lr
import NeuralNetwork as nn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def LRtrain():
    data = []
    for i in range(100):
        x = np.random.uniform(-10., 10.)
        # 采样高斯噪声
        eps = np.random.normal(0., 0.01)
        y = 1.477 * x + 0.089 + eps
        print(x, y)
        data.append([x, y])
    # 数组不支持append，得到了完整的数据才能统一转为数组
    data = np.array(data)
    lr.train(data)
    return None

def NNtrain():
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print("csv文件存储在：",dataset_path)
    # 效能（每加仑能开多少公里），气缸数，排量，马力，重量， 加速度，型号年份，产地
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    # 文件路径，列名列表，用于替换NA/NaN的值，分隔符，是否忽略分隔符后的空白
    dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
    nn.train(dataset)
##

# LRtrain()
NNtrain()


