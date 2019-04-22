import matplotlib.pyplot as plt
import autograd.numpy as np
# import numpy as np
from autograd import grad

theta = np.array([1, -1])
alpha = 0.2
l1 = np.zeros(50)
l2 = np.zeros(50)
l1[0] = theta[0]
l2[0] = theta[1]
'''
def f3(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    c1 = 1 - np.exp(-np.dot((x - a).transpose(), (x - a))) \
         - np.exp(-np.dot(np.dot((x - b).transpose(), B), (x-b))) \
         + 1/10 * np.log(np.linalg.det(np.dot(1/100, np.identity(len(x))) + np.dot(x, x.transpose())))
    return c1
'''

'''
def f3(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    c1 = 1 - np.exp(-np.dot((x - a).transpose(), (x - a))) \
         - np.exp(-np.dot(np.dot((x - b).transpose(), B), (x-b))) \
         + 1/10 * np.log((x[0]**2 + 0.01) * (x[1]**2 + 0.01) + (x[0]*x[1])**2)
    return c1
'''


def f3_grad(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    c1 = 1 - np.exp(-np.dot((x - a).transpose(), (x - a))) \
         - np.exp(-np.dot(np.dot((x - b).transpose(), B), (x-b))) \
         + 1/10 * np.log(np.linalg.det(np.dot(1/100, np.identity(2)) + np.dot(x, x.transpose())))
    return c1

'''
def grad_f3(x):
    grad = np.zeros(2)
    grad[0] = np.exp(-((x[0]-1)**2 + x[1]**2)) * 2 * (x[0] - 1) \
              + np.exp(-(3 * x[0]**2 - 2 * x[0] * x[1] - 2 * x[0] + 6 * x[1] + 3 * x[1]**2 + 3)) * (6 * x[0] - 2 * x[1] - 2)\
              - 20 * x[0] / (100 * x[0]**2 + 100 * x[1]**2 + 1)
    grad[1] = np.exp(-((x[0]-1)**2 + x[1]**2)) * 2 * x[1] \
              + np.exp(-(3 * x[0]**2 - 2 * x[0] * x[1] - 2 * x[0] + 6 * x[1] + 3 * x[1]**2 + 3)) * (6 * x[1] - 2 * x[0] + 6)\
              - 20 * x[1] / (100 * x[0]**2 + 100 * x[1]**2 + 1)
    return grad
'''
grad_f3 = grad(f3_grad)


def f3(x1, x2):
    f = 1 - np.exp(-(x1-1)**2 - x2**2) - \
        np.exp(-(3 * x1**2 - 2 * x1 * x2 - 2 * x1 + 6 * x2 + 3 * x2**2 + 3)) \
        + 0.1 * np.log((x1**2 + 0.01) * (x2**2 + 0.01) - (x1*x2)**2)
    return f


def update_theta(theta, alpha, grad):
    new_theta = np.array(theta) - np.dot(alpha, np.asarray(grad))
    return new_theta


# first calculation
grad = grad_f3(theta)
print(grad)
theta = update_theta(theta, alpha, grad)

# start loop
i = 0
for i in range(0, 49):
    theta = update_theta(theta, alpha, grad)
    grad = grad_f3(theta)
    i += 1
    l1[i] = theta[0]
    l2[i] = theta[0]
    # print('Round %s theta is %s'%(i, theta))

# plot part
x = np.arange(-1.5, 2, 0.05)
y = np.arange(-2, 2, 0.05)

X, Y = np.meshgrid(x, y)
Z = f3(X, Y)
fig1, ax1 = plt.subplots()

CS = ax1.contour(X, Y, Z, levels=np.linspace(-3, 3, 150))
ax1.set_xlabel(r'x')
ax1.set_ylabel(r'y')
ax1.set_title('Contour of f3')
ax1.clabel(CS, inline=1, fontsize=10)

ax1.plot(l1, l2, 'bo')
plt.show()
