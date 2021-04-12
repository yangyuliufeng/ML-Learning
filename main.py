# !/usr/bin/python3

import dataset
import NeuralNetwork as NN
import LinearRegression as lr
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = dataset.random_linear()
lr.train(data)

train_dataset, test_dataset, test_normed_dataframe, test_labels_dataframe = dataset.auto_mpg()
NN.train(train_dataset, test_dataset, test_normed_dataframe, test_labels_dataframe)

