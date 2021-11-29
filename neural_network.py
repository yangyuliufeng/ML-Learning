import time
from abc import ABC

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, applications


def mpg_train(train_dataset, test_normed_dataframe, test_labels_dataframe):
    # 自定义网络类，继承自keras.engine.Model基类
    class Network(keras.Model, ABC):
        # 初始化函数
        def __init__(self):
            super(Network, self).__init__()
            # 创建3个全连接层，分别为64、64、1个节点
            self.fc1 = layers.Dense(64, activation='relu')
            self.fc2 = layers.Dense(64, activation='relu')
            self.fc3 = layers.Dense(1)

        def call(self, inputs):
            # 依次通过3个全连接层
            outputs = self.fc1(inputs)
            outputs = self.fc2(outputs)
            outputs = self.fc3(outputs)
            return outputs

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
        # TODO: cal MAE with all data?
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

    out = model.predict(test_normed_dataframe)
    print(out)

    return None


def mnist_train(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes):
    input_shape = x_train.shape[1:4]

    network = keras.Sequential()
    network.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))
    network.add(layers.MaxPooling2D(pool_size=(2, 2)))
    network.add(layers.Dropout(0.25))
    network.add(layers.Flatten())
    network.add(layers.Dense(128, activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(num_classes, activation='softmax'))

    opt = optimizers.RMSprop(0.001)

    network.compile(loss=losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    network.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test))

    score = network.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def resnet50_train(x_batch):
    resnet = applications.ResNet50(weights='imagenet', include_top=False)

    out = resnet(x_batch)
    print(out.shape)

    # 新建池化层降维，shape由[batch_size,7,7,2048]变为[batch_size,1,1,2048]，删减维度后变为[batch_size,2048]
    global_average_layer = layers.GlobalAveragePooling2D()
    out = global_average_layer(out)
    print(out.shape)

    # 新建全连接层，设置输出节点数为100
    fc = layers.Dense(100)
    out = fc(out)
    print(out.shape)

    network = keras.Sequential([resnet, global_average_layer, fc])

    out = network.predict(x_batch)
    print(out.shape)


def benchmark(x_train, y_train, epochs):
    # 从keras.applications中获取'ResNet50'属性
    model = getattr(applications, 'ResNet50')(weights=None)
    # 优化器
    optimizer = tf.optimizers.SGD(0.01)

    # 速度
    sec_per_epoch = []
    img_per_sec = []

    batches_per_epoch = len(x_train)

    def train_one_batch(batch, target):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss = tf.losses.sparse_categorical_crossentropy(target, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for current_epoch in range(epochs):
        time1 = time.time()
        if current_epoch == 0:
            print('Starting warmup...')
        for current_iter in range(batches_per_epoch):
            train_one_batch(x_train[current_iter - 1], y_train[current_iter - 1])

        time2 = time.time()
        time_cost = time2 - time1
        sec_per_epoch.append(time_cost)
        # 打印本epoch的耗时、速率
        print('Epoch #%d: cost %.1f sec' % (current_epoch, time_cost))
        print()


        img_sec = x_train[0].shape[0] * batches_per_epoch / time_cost
        print('Epoch #%d: %.1f img/sec' % (current_epoch, img_sec))
        # img_per_sec.append(img_sec)
