# !/usr/bin/python3

import tensorflow as tf
import numpy as np
import os


# import PythonTest as pt
# import TensorflowTest as tt
# import LinearRegression as lr
import NeuralNetwork as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print(pt.grammar(5))
# tt.tensorflow(2,3)
nn.train()

'''
def train():
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
train()
'''
