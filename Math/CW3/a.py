import numpy as np
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
alpha = 0.4246
beta = 0.4692
Phi = np.array([[1, 2],
                [1, 3]])
Y = np.array([[- 0.75426779],
              [-0.5480492]])


def lml(alpha, beta, Phi, Y):
    (n, m) = Phi.shape
    func = - 0.5 * n * np.log(2 * np.pi) \
           - 0.5 * np.log(np.linalg.det(alpha * np.dot(np.dot(Phi, np.identity(m)),np.transpose(Phi)) + beta * np.identity(n)))  \
           - 0.5 * np.dot(np.dot(np.transpose(Y),np.linalg.inv(alpha * np.dot(np.dot(Phi, np.identity(m)),np.transpose(Phi)) + beta * np.identity(n))),Y)
    return func.item(0)

def grad_lml(alpha, beta, Phi, Y):

    (n, m) = Phi.shape
    k = alpha * np.dot(np.dot(Phi, np.identity(m)), np.transpose(Phi)) + beta * np.identity(n)
    l = np.dot(Phi,np.transpose(Phi))
    k1 = np.dot(np.linalg.inv(k),l)
    grad1 = - 0.5 * np.trace(np.dot(np.linalg.inv(k),l)) + 0.5 * np.dot(np.dot(np.transpose(Y),k1),np.dot(np.linalg.inv(k),Y))
    
    grad2 = - 0.5 * np.trace(np.linalg.inv(k)) + 0.5 * np.dot(np.dot(np.transpose(Y),np.linalg.inv(k)),np.dot(np.linalg.inv(k),Y))

    return np.array([grad1.item(0), grad2.item(0)])


print(lml(alpha, beta, Phi, Y))

grad = grad_lml(alpha, beta, Phi, Y)
print(grad)



'''

    '''
