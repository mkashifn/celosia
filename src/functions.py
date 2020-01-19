#!/bin/python
import numpy as np

class Function:
  def __init__(self):
    pass
  def __call__(self, x):
    return self.fx(x)
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
    exps = np.exp(x - np.max(x))
    sm = exps / np.sum(exps)
    return sm

  def dfx(self, fx):
    s = fx.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

class Relu(Function):
  # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
  def fx(self, x):
    return x * (x > 0)

  def dfx(self, fx):
    return 1. * (fx > 0)

sigmoid = Logistic()
softmax = Softmax()
relu = Relu()