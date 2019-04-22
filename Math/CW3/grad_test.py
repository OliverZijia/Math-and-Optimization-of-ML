# import numpy as np
import autograd.numpy as np
from autograd import grad


def f1(x):  # Taylor approximation to sine function
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    # c1 = np.dot(theta.transpose(), theta)
    c1 = np.dot(x.transpose(), x) + np.dot(x.transpose(), (np.dot(B, x))) - np.dot(a.transpose(), x) + np.dot(b.transpose(), x)
    return c1

def grad_f1(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot(2, x).transpose() + np.dot(2, np.dot(B, x)).transpose() - a.transpose() + b.transpose()
    return grad

def f2(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    # c1 = np.dot(theta.transpose(), theta)
    c1 = np.sin(np.dot((x-a).transpose(), (x-a))) + np.dot((x-b).transpose(), (np.dot(B, (x-b))))
    return c1


def grad_f2(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot((np.cos(np.dot((x-a).transpose(), (x-a)))), (x-a).transpose()) + np.dot(2, np.dot((x-b).transpose(), B))
    return grad


def f3(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    c1 = 1 - np.exp(-np.dot((x - a).transpose(), (x - a))) \
         - np.exp(-np.dot(np.dot((x - b).transpose(), B), (x-b))) \
         + 1/10 * np.log(np.linalg.det(np.dot(1/100, np.identity(2)) + np.dot(x, x.transpose())))
    return c1


def grad_f3(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot((np.exp(-np.dot((x-a).transpose(), (x-a)))), (x-a).transpose()) \
           - np.dot(np.exp(-np.dot((x-b).transpose(), np.dot(B, (x-b)))), np.dot(2, np.dot((x-b).transpose(), B))) \
           + np.dot((1/10/(1/100 + np.dot(x, x.transpose()))), np.dot(2, x.transpose()))
    return grad


theta = np.array([[1], [-1]])
gradf1 = grad(f1)
gradf2 = grad(f2)
gradf3 = grad(f3)
print( "AutoGradient of f1 is", gradf1(theta))
print( "ManGradient of f1 is", grad_f1(theta))
print( "AutoGradient of f2 is", gradf2(theta))
print( "ManGradient of f2 is", grad_f2(theta))
print( "AutoGradient of f3 is", gradf3(theta))
#print( "ManGradient of f3 is", grad_f3(theta))









'''
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad


def taylor_sine(x):  # Taylor approximation to sine function
    ans = currterm = x
    i = 0
    while np.abs(currterm) > 0.001:
        currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans

grad_sine = grad(taylor_sine)
print ("Gradient of sin(pi) is", grad_sine(np.pi))
'''
