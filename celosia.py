#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object
import math

class Celosia:
  def __init__(self):
    pass

  def max_nh(self, ni, no, ns):
    '''a rule-of-thumb for the upper bound of the number of neurons in hidden layers:
       Ref: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
       where ni and no = number of input and outputs and,
       ns = number of samples in training data set.'''
    alpha = 2
    nh = math.ceil(ns/(alpha * (ni + no)))
    return nh

  def create_network_sigmoid(self, i, o, lh, eta=0.5):
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

  def create_optimal_network(self, inputs, outputs, N, epochs = 10000, hmax=5, nmax=5):
    '''create an optimal network by trying different structures.
       Parameters: inputs, outputs = input and output vectors.
                   N = number of different structures to try.
                   epochs = number of epochs to try for each structure, default = 10000.
                   hmax = maximum number of hidden layers, default = 5.
                   hmax = maximum number of neurons in a hidden layer, default = 5.'''
    #nmax = max_nh(i, o, inputs.shape[0])
    lnn = [] # list of neural networks
    le = []  # list of errors
    i = inputs.shape[1] # number of colums in the input
    o = outputs.shape[1] # number of colums in the output
    for j in range(N):
      lh = [] # list of the number of neurons in a hidden layer
      h = randint(1, hmax)
      for k in range(h):
        lh.append(randint(1, nmax))
      nn = self.create_network_sigmoid(i, o, lh)
      lnn.append(nn)
      e = nn.train(inputs, outputs, epochs)
      le.append(e)
      print ("structure-{}: error: {}".format(j+1, e))
      nn.draw(inputs, outputs, file="structure-{}".format(j+1), cleanup=True)
    print le
    mi = le.index(min(le)) # minimum
    print ("Minimum error index: {}".format(mi))
    lnn[mi].draw(inputs, outputs, file="most-optimal", cleanup=True)
    return lnn[mi]
