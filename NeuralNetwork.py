# !/usr/bin/python3

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 在线下载汽车效能数据集
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print("csv文件存储在",dataset_path)

# 效能（每加仑能开多少公里），气缸数，排量，马力，重量， 加速度，型号年份，产地
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
# 文件路径，列名列表，用于替换NA/NaN的值，分隔符，是否忽略分隔符后的空白
dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

# %% 处理数据

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

# ???
sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")

# 查看训练集的输入X的统计数据
train_stats = train_dataset.describe()
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

# 构建Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
# 随机打散，批量化
train_db = train_db.shuffle(100).batch(32)

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
print(model.summary())
# 创建优化器，指定学习率
optimizer = tf.keras.optimizers.RMSprop(0.001)

# # 未训练时测试
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# example_result

train_mae_losses = []
test_mae_losses = []

for epoch in range(200):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            # MSE：均方误差
            loss = tf.reduce_mean(losses.MSE(y, out))
            # MAE：平均绝对误差
            mae_loss = tf.reduce_mean(losses.MAE(y, out))    
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(losses.MAE(test_labels, out)))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses, label='Train')
plt.plot(test_mae_losses, label='Test')
plt.legend()

# plt.ylim([0,10])
plt.legend()
plt.savefig('auto.svg')
plt.show()

