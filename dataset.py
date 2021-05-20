# !/usr/bin/python3

import keras
import pandas
import tensorflow as tf
import numpy


def random_linear():
    data = []
    for i in range(100):
        x = numpy.random.uniform(-10., 10.)
        # 采样高斯噪声
        eps = numpy.random.normal(0., 0.01)
        y = 1.477 * x + 0.089 + eps
        data.append([x, y])
    # 数组不支持append，得到了完整的数据才能将列表 转为数组
    data = numpy.array(data)
    return data


def auto_mpg():
    file_name = "auto-mpg.data"
    dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset_path = keras.utils.get_file(file_name, dataset_url)
    print("csv文件存储在：", dataset_path)

    # 效能（每加仑能开多少公里），气缸数，排量，马力，重量， 加速度，型号年份，产地
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    # 文件路径，列名列表，用于替换NA/NaN的值，分隔符，是否忽略分隔符后的空白
    dataframe = pandas.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ",
                                skipinitialspace=True)
    # 删除空白数据
    dataframe = dataframe.dropna()

    # 取出origin列，并根据其写入新列
    origin = dataframe.pop('Origin')
    dataframe['USA'] = (origin == 1) * 1.0
    dataframe['Europe'] = (origin == 2) * 1.0
    dataframe['Japan'] = (origin == 3) * 1.0

    # 切分为训练集和测试集
    train_dataframe = dataframe.sample(frac=0.8, random_state=0)
    test_dataframe = dataframe.drop(train_dataframe.index)

    # 展现气缸、排量、马力、重量四个变量两两间的关系
    # sns.pairplot(train_dataframe[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")

    # 查看训练集的输入X的统计数据
    train_stats_dataframe = train_dataframe.describe()
    # 移除MPG字段
    train_stats_dataframe.pop("MPG")
    # 旋转矩阵
    train_stats_dataframe = train_stats_dataframe.transpose()

    # 将MPG字段移出为标签数据：
    train_labels_dataframe = train_dataframe.pop('MPG')
    test_labels_dataframe = test_dataframe.pop('MPG')

    # 根据均值和标准差，完成数据的标准化
    def norm(x):
        return (x - train_stats_dataframe['mean']) / train_stats_dataframe['std']

    train_normed_dataframe = norm(train_dataframe)
    test_normed_dataframe = norm(test_dataframe)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_normed_dataframe.values, train_labels_dataframe.values))
    train_dataset = train_dataset.shuffle(100).batch(32)

    return train_dataset, test_normed_dataframe, test_labels_dataframe


def mnist():
    num_classes = 10
    # 输入图片的长宽
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 类别标签转换为OneHot编码
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes
