# !/usr/bin/python3
from abc import ABC

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, losses, engine, optimizers


def mpg_train(train_dataset, test_normed_dataframe, test_labels_dataframe):
    # 自定义网络类，继承自keras.Model基类
    class Network(engine.Model, ABC):
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

    for epoch in range(10):
        for step, (x, y) in enumerate(train_dataset):
            # 一次训练32个数据
            with tf.GradientTape() as tape:
                out = model(x)
                # MSE：均方误差
                train_mse_loss = tf.reduce_mean(losses.MSE(y, out))
            grads = tape.gradient(train_mse_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # TODO: cal MAE with all data
        # MAE：平均绝对误差
        train_mae_loss = tf.reduce_mean(losses.MAE(y, out))
        train_mae_losses.append(float(train_mae_loss))
        out = model(tf.constant(test_normed_dataframe.values))
        test_mae_loss = tf.reduce_mean(losses.MAE(test_labels_dataframe, out))
        test_mae_losses.append(float(test_mae_loss))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(train_mae_losses, label='Train')
    plt.plot(test_mae_losses, label='Test')
    # 设置坐标轴范围
    plt.ylim([0, 200])
    plt.ylim([0, 30])
    # 给图像加图例
    plt.legend()

    plt.savefig('auto.svg')
    plt.show()

    out = model.predict(test_normed_dataframe)  # 模型预测，预测结果保存在out中
    print(out)

    return None


def mnist_train(x_train, y_train, x_test, y_test, input_shape, num_classes):
    batch_size = 128
    epochs = 2

    model = keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    opt = optimizers.RMSprop(0.001)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

