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

sigmoid = Logistic()