import argparse
import os
import timeit
import numpy
import tensorflow as tf
from tensorflow.keras import applications

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 参数
batch_size = 4
batches_to_warmup = 1
batches_per_iter = 2
iter_num = 2


# 学习率
lr = 0.01
# 从keras.applications中获取'ResNet50'属性
model = getattr(applications, 'ResNet50')(weights=None)
# 优化器
optimizer = tf.optimizers.SGD(lr)

# 数据集
data = tf.random.uniform([batch_size, 224, 224, 3])
target = tf.random.uniform([batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

# 结果
img_per_sec = []
sec_per_iter = []


def log(s, nl=True):
    print(s, end='\n' if nl else '')


def train_one_batch():
    with tf.GradientTape() as tape:
        out = model(data, training=True)
        loss = tf.losses.sparse_categorical_crossentropy(target, out)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def run_benchmark():
    for current_iter in range(iter_num):
        if current_iter == 0:
            log('Starting warmup...')
            timeit.timeit(lambda: train_one_batch(), number=batches_to_warmup)
            log('Running benchmark...')
        # 重复调用benchmark_step(state)函数batches_per_iter次，并计算时间
        time = timeit.timeit(lambda: train_one_batch(), number=batches_per_iter)
        sec_per_iter.append(time)
        # 打印本iter的耗时、速率
        log('Iter #%d: cost %.1f sec' % (current_iter, time))
        img_sec = batch_size * batches_per_iter / time
        log('Iter #%d: %.1f img/sec' % (current_iter, img_sec))
        img_per_sec.append(img_sec)


run_benchmark()

sec_per_iter_mean = numpy.mean(sec_per_iter)
sec_per_iter_conf = 1.96 * numpy.std(sec_per_iter)
log('sec/iter : %.1f +-%.1f' % (sec_per_iter_mean, sec_per_iter_conf))

img_per_sec_mean = numpy.mean(img_per_sec)
img_per_sec_conf = 1.96 * numpy.std(img_per_sec)
log('img/sec : %.1f +-%.1f' % (img_per_sec_mean, img_per_sec_conf))