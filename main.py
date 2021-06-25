# !/usr/bin/python3

import my_dataset
import neural_network
import linear_regression
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if False:
    data = my_dataset.random_linear()
    linear_regression.train(data)

if False:
    train_dataset, test_normed_dataframe, test_labels_dataframe = my_dataset.auto_mpg()
    neural_network.mpg_train(train_dataset, test_normed_dataframe, test_labels_dataframe)

if False:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = my_dataset.mnist()
    neural_network.mnist_train(x_train, y_train, x_test, y_test, input_shape, num_classes)

if True:
    x_batch = my_dataset.synthetic_batch(4)
    neural_network.resnet50_train(x_batch)

