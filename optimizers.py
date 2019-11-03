#!/bin/python
import numpy as np

class Optimizer:
  def __init__(self):
    self.cfg={}

  def __call__(self, vt, theta, gt, eta):
    return (vt, theta)

class NoOptimizer(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, vt, theta, gt, eta):
    vt = vt
    theta = theta - gt * eta
    return (vt, theta)

class Momentum(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, vt, theta, gt, eta):
    gamma = self.cfg.get('gamma', 0.5)
    vt = (gamma * vt) + (gt * eta)
    theta = theta - vt
    return (vt, theta)

class Adam(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, vt, theta, gt, eta):
    beta1 = self.cfg.get('beta1', 0.9)
    beta2 = self.cfg.get('beta2', 0.9999)
    eps = self.cfg.get('eps', 1e-8)
    mt = (beta1 * vt) + (1 - beta1)*gt
    mt_h = mt / (1 - beta1)
    vt = (beta2 * vt) + (1 - beta2)*(gt**2)
    vt_h = vt / (1 - beta2)
    theta = theta - (eta / (np.sqrt(vt_h) + eps))*mt_h
    return (vt, theta)

no_optimizer = NoOptimizer()
momentum_default = Momentum()
adam_default = Adam()