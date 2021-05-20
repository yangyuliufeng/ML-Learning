# !/usr/bin/python3

import dataset
import NeuralNetwork as NN
import LinearRegression as lr
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data = dataset.random_linear()
# lr.train(data)

# train_dataset, test_normed_dataframe, test_labels_dataframe = dataset.auto_mpg()
# NN.mpg_train(train_dataset, test_normed_dataframe, test_labels_dataframe)

# (x_train, y_train), (x_test, y_test), input_shape, num_classes = dataset.mnist()
# NN.mnist_train(x_train, y_train, x_test, y_test, input_shape, num_classes)

NN.resnet50_train()

