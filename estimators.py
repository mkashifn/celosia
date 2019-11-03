#!/bin/python
import numpy as np
from functions import softmax

class Estimator:
  def __init__(self):
    pass
  def __call__(self, A, B):
    return self.fx(A, B)
  def fx(self, A, B):
    return 0
  def dfx(self, A, B):
    return 0

class MSE(Estimator):
  def __init__(self):
    pass
  def fx(self, A, B):
    return np.square(np.subtract(A, B)).mean()
  def dfx(self, A, B):
    n = A.shape[1] #number of columns
    return (2*np.subtract(A, B))/n

class CrossEntropy(Estimator):
  def fx(self, A, B):
    '''A = predicted output, B is target output.'''
    B = B.argmax(axis=1)
    m = B.shape[0]
    p = softmax(A)
    log_likelihood = -np.log(p[range(m),B])
    return np.sum(log_likelihood) / m
  def dfx(self, A, B):
    B = B.argmax(axis=1)
    m = B.shape[0]
    p = softmax(A)
    p[range(m),B] -= 1
    return p/m


mse = MSE()
cross_entropy = CrossEntropy()