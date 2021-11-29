import numpy as np


def train(data):
    lr = 0.01  # 学习率
    initial_w = initial_b = 0
    num_iterations = 1000  # 训练优化次数
    [w, b] = gradient_descent(data, initial_w, initial_b, lr, num_iterations)
    loss = mse(w, b, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')
    return None


# 计算最优解b、w上的均方差
def mse(w, b, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(w_current, b_current, points, lr):
    new_w = w_current
    new_b = b_current
    w_gradient = b_gradient = 0
    m = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对w的导数grad_w = 2(wx+b-y)*x
        w_gradient += (2 / m) * x * ((w_current * x + b_current) - y)
        # 误差函数对b的导数grad_b = 2(wx+b-y)
        b_gradient += (2 / m) * ((w_current * x + b_current) - y)
        # 根据学习率lr更新w和b
        new_w = w_current - (lr * w_gradient)
        new_b = b_current - (lr * b_gradient)

    return [new_w, new_b]


def gradient_descent(points, starting_w, starting_b, lr, num_iterations):
    w = starting_w
    b = starting_b
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并更新一次
        [w, b] = step_gradient(w, b, np.array(points), lr)
        loss = mse(w, b, points)  # 计算当前的均方差，用于监控训练进度
        if step % 50 == 0:  # 打印误差和实时的w,b 值
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [w, b]  # 返回最后一次的w和b
