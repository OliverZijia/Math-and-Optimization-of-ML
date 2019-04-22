# 暂未画图，未加入50次循环停止条件，无grad_f3
import numpy as np

theta = [1, -1]
alpha = 0.2
tol_L = 0.1


def grad_f1(x):
    grad = [0, 0]
    grad[0] = 8 * x[0] - 2 * x[1] - 1
    grad[1] = 8 * x[1] - 2 * x[0] - 1
    return grad


def grad_f2(x):
    grad = [0, 0]
    grad[0] = np.cos((x[0] - 1) ** 2 + x[1] ** 2) * 2 * (x[0] - 1) + 6 * x[0] - 2 * x[1] - 2
    grad[1] = np.cos((x[0] - 1) ** 2 + x[1] ** 2) * 2 * x[1] + 6 * x[1] - 2 * x[0] + 6
    return grad


def update_theta(theta, alpha, grad):
    new_theta = np.array(theta) - 0.1 * np.asarray(grad)
    return new_theta


# first calculation
grad = grad_f1(theta)
theta = update_theta(theta, alpha, grad)

# start recursion
i = 1
for i in range(1, 51):
    theta = update_theta(theta, alpha, grad)
    grad = grad_f1(theta)
    i += 1
    print('Round %s theta is %s'%(i, theta))