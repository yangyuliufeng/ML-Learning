# !/usr/bin/python3

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

def train(raw_dataset):

    dataset= raw_dataset.copy()
    # 清除空白数据
    dataset = dataset.dropna()

    # 取出origin列，并根据其写入新列
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    # 切分为训练集和测试集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # 展现气缸、排量、马力、重量四个变量两两间的关系
    sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")

    # 查看训练集的输入X的统计数据
    train_stats = train_dataset.describe()
    # 移除MPG字段
    train_stats.pop("MPG")
    # 旋转矩阵
    train_stats = train_stats.transpose()
    
    # 将MPG字段移出为标签数据：
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # 根据均值和标准差，完成数据的标准化
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # 构建tf Dataset对象
    train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
    train_db = train_db.shuffle(100).batch(32)

    test_db = tf.data.Dataset.from_tensor_slices((normed_test_data.values, test_labels.values))
    test_db = test_db.shuffle(100).batch(32)

    # 自定义网络类，继承自keras.Model基类
    class Network(keras.Model):
        # 初始化函数
        def __init__(self):
            super(Network, self).__init__()
            # 创建3个全连接层，分别为64、64、1个节点
            self.fc1 = layers.Dense(64, activation='relu')
            self.fc2 = layers.Dense(64, activation='relu')
            self.fc3 = layers.Dense(1)

        def call(self, inputs, training=None, mask=None):
            # 依次通过3个全连接层
            x = self.fc1(inputs)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    # 创建网络类实例
    model = Network()
    # 通过build函数完成内部张量的创建，其中4为任意设置的batch数量，9为输入特征长度
    model.build(input_shape=(None, 9))

    # 创建优化器，指定学习率
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    train_mae_losses = []
    test_mae_losses = []

    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            # 一次训练32个数据
            with tf.GradientTape() as tape:
                out = model(x)
                # MSE：均方误差
                train_mse_loss = tf.reduce_mean(losses.MSE(y, out))
                # MAE：平均绝对误差
                train_mae_loss = tf.reduce_mean(losses.MAE(y, out))
            grads = tape.gradient(train_mse_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_mae_losses.append(float(train_mae_loss))
        out = model(tf.constant(normed_test_data.values))
        test_mae_loss = tf.reduce_mean(losses.MAE(test_labels,out))
        test_mae_losses.append(float(test_mae_loss))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(train_mae_losses, label='Train')
    plt.plot(test_mae_losses, label='Test')
    #设置坐标轴范围
    plt.ylim([0, 200])
    plt.ylim([0, 30])
    #给图像加图例
    plt.legend()

    plt.savefig('auto.svg')
    plt.show()

    x, y = next(iter(test_db))  # 加载一个batch的测试数据
    print(x.shape)  # 打印当前batch的形状
    out = model.predict(x)  # 模型预测，预测结果保存在out中
    print(out)

    return None