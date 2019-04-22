# 暂未画图，未加入50次循环停止条件，无grad_f3
import matplotlib.pyplot as plt
import numpy as np

theta = np.array([1, -1])
alpha = 0.03


def f2(x1, x2):
    f = 1 - np.exp(-(x1-1)**2 - x2**2) - \
        np.exp(-(3 * x1**2 - 2 * x1 * x2 - 2 * x1 + 6 * x2 + 3 * x2**2 + 3)) \
        + 0.1 * np.log((x1**2 + 0.01) * (x2**2 + 0.01) - (x1*x2)**2)
    return f

'''
def grad_f2(x):
    B = np.array([[3, -1], [-1, 3]])
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    grad = np.dot((np.cos(np.dot((x-a).transpose(), (x-a)))), (x-a).transpose()) \
           + np.dot(2, np.dot((x-b).transpose(), B))
    return grad
'''


def grad_f2(x):
    grad = np.zeros(2)
    grad[0] = np.exp(-((x[0]-1)**2 + x[1]**2)) * 2 * (x[0] - 1) \
              + np.exp(-(3 * x[0]**2 - 2 * x[0] * x[1] - 2 * x[0] + 6 * x[1] + 3 * x[1]**2 + 3)) * (6 * x[0] - 2 * x[1] - 2)\
              - 20 * x[0] / (100 * x[0]**2 + 100 * x[1]**2 + 1)
    grad[1] = np.exp(-((x[0]-1)**2 + x[1]**2)) * 2 * x[1] \
              + np.exp(-(3 * x[0]**2 - 2 * x[0] * x[1] - 2 * x[0] + 6 * x[1] + 3 * x[1]**2 + 3)) * (6 * x[1] - 2 * x[0] + 6)\
              - 20 * x[1] / (100 * x[0]**2 + 100 * x[1]**2 + 1)
    return grad


def update_theta(theta, alpha, grad):
    new_theta = np.array(theta) - alpha * np.asarray(grad)
    return new_theta


# first calculation
grad = grad_f2(theta)
theta = update_theta(theta, alpha, grad)
l1 = np.zeros(50)
l2 = np.zeros(50)
l1[0] = theta[0]
l2[0] = theta[1]
# start loop
i = 0
for i in range(0, 49):
    theta = update_theta(theta, alpha, grad)
    grad = grad_f2(theta)
    i += 1
    l1[i] = theta[0]
    l2[i] = theta[1]
#    print('Round %s theta is %s'%(i, theta))
print(l1)
# plot part

x = np.arange(-1, 1, 0.05)
y = np.arange(-2, 0.5, 0.05)

X, Y = np.meshgrid(x, y)
Z = f2(X, Y)
fig1, ax1 = plt.subplots()

CS = ax1.contour(X, Y, Z, levels=np.linspace(-12.5, 12.5, 50))
ax1.set_xlabel(r'x')
ax1.set_ylabel(r'y')
ax1.set_title('Contour of f2')
ax1.clabel(CS, inline=1, fontsize=10)

ax1.plot(l1, l2, 'bo')
plt.show()