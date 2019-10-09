#!/bin/python
import numpy as np

class Function:
  def __init__(self):
    pass

  def fx(self, x):
    return x

  def dfx(self, x):
    return np.zeros(x.shape)

class Logistic(Function):
  def __init__(self):
    pass

  def fx(self, x):
    return 1/(1+np.exp(-x))

  def dfx(self, fx):
    return fx * (1 - fx)

class Softmax(Function):
  #https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
  def __init__(self):
    pass

  def fx(self, x):
    z = x - np.max(x)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm

  def dfx(self, fx):
    s = fx.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

sigmoid = Logistic()
softmax = Softmax()