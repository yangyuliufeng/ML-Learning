# !/usr/bin/python3
import tensorflow as tf


def tensorflow(parameter1,parameter2):
    print("tensorflow的版本是：", tf.__version__)
    print("tensorflow的路径是：", tf.__path__)




    default_g = tf.compat.v1.get_default_graph()

    a = tf.constant(parameter1)
    b = tf.constant(parameter2)
    print(a)
    with tf.GradientTape() as tape:  # 构建梯度环境
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        tape.watch([a])  # 将a加入梯度跟踪列表
        y = b * a ** 2
        [dy_da] = tape.gradient(y, [a])
        print(dy_da)

    tensor1 = tf.constant(value=[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                          dtype=tf.int32,
                          shape=[4, 4],
                          name='tensor1')
    print(tensor1)

    return None
