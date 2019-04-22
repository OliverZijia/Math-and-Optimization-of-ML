# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

import numpy as np

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
