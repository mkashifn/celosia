#!/bin/python
import numpy as np

class Estimator:
  def __init__(self):
    pass
  def __call__(self, A, B):
    return 0

class MSE(Estimator):
  def __init__(self):
    pass
  def __call__(self, A, B):
    #print ("A:", A, "B:", B, "Diff:", np.subtract(A, B), "SqDiff:", np.square(np.subtract(A, B)), "MSE:", np.square(np.subtract(A, B)).mean())
    return np.square(np.subtract(A, B)).mean()
  def fx(self, A, B):
    return np.square(np.subtract(A, B)).mean()
  def dfx(self, A, B):
    n = A.shape[1] #number of columns
    return (2*np.subtract(A, B))/n

mse = MSE()