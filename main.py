import my_dataset
import neural_network
import linear_regression
import os
import configparser
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("tensorflow的版本是：", tf.__version__)
print("tensorflow的路径是：", tf.__path__)

cf = configparser.ConfigParser()
cf.read("config.ini")
secs = cf.sections()


if cf.get("Linear-Regression", "random-linger") == "true":
    data = my_dataset.random_linear()
    linear_regression.train(data)


if cf.get("Neural-Network", "mpg-train") == "true":
    train_dataset, test_normed_dataframe, test_labels_dataframe = my_dataset.auto_mpg()
    neural_network.mpg_train(train_dataset, test_normed_dataframe, test_labels_dataframe)


if cf.get("Neural-Network", "mnist-train") == "true":
    batch_size = 4
    epochs = 2
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = my_dataset.mnist(num_classes)
    neural_network.mnist_train(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes)


if cf.get("Neural-Network", "resnet50-train") == "true":
    batch_size = 4
    x_batch = my_dataset.synthetic_batch(batch_size)
    neural_network.resnet50_train(x_batch)

if cf.get("Neural-Network", "benchmark-train") == "true":
    batch_size = 4
    batches_per_epoch = 100
    epochs = 10
    x_train, y_train = my_dataset.synthetic_epoch(batch_size, batches_per_epoch)
    neural_network.benchmark(x_train, y_train, epochs)
