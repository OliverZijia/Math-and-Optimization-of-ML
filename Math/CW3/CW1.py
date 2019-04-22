import numpy as np


def grad_f1(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot(2, x).transpose() + np.dot(2, np.dot(B, x)).transpose() - a.transpose() + b.transpose()
    return grad


def grad_f2(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot((np.cos(np.dot((x-a).transpose(), (x-a)))), (x-a).transpose()) \
           + np.dot(2, np.dot((x-b).transpose(), B))
    return grad
