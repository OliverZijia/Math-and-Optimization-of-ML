# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np
    
def grad_f1(x):
    grad = [0, 0]
    grad[0] = 8 * x[0] - 2 * x[1] - 1
    grad[1] = 8 * x[1] - 2 * x[0] - 1
    return np.array([grad[0],grad[1]])

def grad_f2(x):
    grad = [0, 0]
    grad[0] = np.cos((x[0] - 1) ** 2 + x[1] ** 2) * 2 * (x[0] - 1) + 6 * x[0] - 2 * x[1] - 2
    grad[1] = np.cos((x[0] - 1) ** 2 + x[1] ** 2) * 2 * x[1] + 6 * x[1] - 2 * x[0] + 6
    return np.array([grad[0],grad[1]])

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    pass

print(grad_f1([1,1]))
print(grad_f2([1,1]))
