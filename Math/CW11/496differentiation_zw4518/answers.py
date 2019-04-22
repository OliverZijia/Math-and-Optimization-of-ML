# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

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

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    pass


