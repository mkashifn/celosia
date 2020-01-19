#!/bin/python
import numpy as np

class Optimizer:
  def __init__(self):
    self.cfg={}

  def __call__(self, grads, theta, gt, eta):
    return (grads, theta)

class NoOptimizer(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, grads, theta, gt, eta):
    theta = theta - gt * eta
    return (grads, theta)

class Momentum(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, grads, theta, gt, eta):
    vt_1 = grads.get('vt', theta * 0)
    gamma = self.cfg.get('gamma', 0.5)
    vt = (gamma * vt_1) + (gt * eta)
    theta = theta - vt
    grads['vt'] = vt
    return (grads, theta)

class Adam(Optimizer):
  def __init__(self, cfg={}):
    self.cfg=cfg

  def __call__(self, grads, theta, gt, eta):
    vt_1 = grads.get('vt', theta * 0)
    mt_1 = grads.get('mt', theta * 0)
    beta1 = self.cfg.get('beta1', 0.9)
    beta2 = self.cfg.get('beta2', 0.9999)
    eps = self.cfg.get('eps', 1e-8)
    mt = (beta1 * mt_1) + (1 - beta1)*gt
    mt_h = mt / (1 - beta1)
    vt = (beta2 * vt_1) + (1 - beta2)*(gt**2)
    vt_h = vt / (1 - beta2) # Adam
    vt_h = np.maximum(vt_1, vt) # AMSGrad
    theta = theta - (eta / (np.sqrt(vt_h) + eps))*mt_h
    grads['vt'] = vt
    grads['mt'] = mt
    return (grads, theta)

no_optimizer = NoOptimizer()
momentum_default = Momentum()
adam_default = Adam()