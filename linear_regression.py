# !/usr/bin/python3

import numpy as np


def train(data):
    lr = 0.01  # 学习率
    initial_b = initial_w = 0
    num_iterations = 1000  # 训练优化次数
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')
    return None


# 计算最优解b、w上的均方差
def mse(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, lr):
    new_b = b_current
    new_w = w_current
    b_gradient = w_gradient = 0
    m = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对b的导数grad_b = 2(wx+b-y)
        b_gradient += (2 / m) * ((w_current * x + b_current) - y)
        # 误差函数对w的导数grad_w = 2(wx+b-y)*x
        w_gradient += (2 / m) * x * ((w_current * x + b_current) - y)
        # 根据学习率lr更新b和w
        new_b = b_current - (lr * b_gradient)
        new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并更新一次
        [b, w] = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)  # 计算当前的均方差，用于监控训练进度
        if step % 50 == 0:  # 打印误差和实时的w,b 值
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w]  # 返回最后一次的w,b
