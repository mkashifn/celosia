#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object
import math

def max_nh(ni, no, ns):
  '''a rule-of-thumb for the upper bound of the number of neurons in hidden layers:
     Ref: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
     where ni and no = number of input and outputs and,
     ns = number of samples in training data set.'''
  alpha = 2
  nh = math.ceil(ns/(alpha * (ni + no)))
  return nh

def create_network_sigmoid(i, o, lh, eta=0.5):
  '''Create a neural network by making use of sigmoid activation function in all layers.
     Parameters: i,o =  number of neurons in [input|output] layers.
                 lh = list containing the number of neurons in each hidden layer.
                 eta = learning rate, 0.5 by default.'''
  assert type(i) is int, "parameter i (number of input neurons) needs to be an integer"
  assert type(o) is int, "parameter o (number of output neurons) needs to be an integer"
  assert type(lh) is list, "parameter h (list of neurons in each hidden layer) needs to be a list"
  assert len(lh) >= 1, "there needs to be at least one hidden layer"

  w = None # None means randomly initialize weights
  nn = Sequential(mse, eta)
  # input layer
  nn.add_layer(lh[0], sigmoid, 0.0, w, i)
  # hidden layers
  for h in lh[1:]:
    nn.add_layer(h, sigmoid, 0.0, w)
  # output layer
  nn.add_layer(o, sigmoid, 0.0, w)
  return nn

def create_optimal_network(i, o, n, inputs, outputs, epochs = 10000, mh=5):
  '''create an optimal network by trying different structures.
     Parameters: i,o = number of neurons in [input|output] layers.
                 n = number of different structures to try.
                 inputs, outputs = input and output vectors.
                 epochs = number of epochs to try for each structure, default = 10000.
                 mh = maximum number of hidden layers, default = 5.'''
  nh = max_nh(i, o, inputs.shape[0])
  lnn = [] # list of neural networks
  le = []  # list of errors
  for j in range(n):
    lh = [] # list of the number of neurons in a hidden layer
    nh = randint(1, mh)
    for k in range(nh):
      lh.append(randint(1, nh))
    nn = create_network_sigmoid(i, o, lh)
    lnn.append(nn)
    e = nn.train(inputs, outputs, epochs)
    le.append(e)
    print ("try-{}: error: {}".format(j, e))
    nn.draw(inputs, outputs, file="iteration-{}".format(j+1), cleanup=True)
  print le
  mi = le.index(min(le)) # minimum
  print mi
  lnn[mi].draw(inputs, outputs, file="most-optimal", cleanup=True)
