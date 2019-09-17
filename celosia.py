#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object
import math
from sklearn.model_selection import train_test_split
import time
from multiprocessing import Process, Queue

def evaluate_nn(nn, epochs, X_train, X_test, y_train, y_test, mp, q):
  e_ratio = 0
  total_epochs = 0
  while (e_ratio < 1.0) and (total_epochs < (10 * epochs)):
    total_epochs += epochs
    e_tr = nn.train(X_train, y_train, epochs) # training loss
    e_tst = mse(y_test, nn.output(X_test)) # test loss
    e_ratio = e_tst / e_tr
  res = {nn.name:{'tl': e_tr, 'vl': e_tst, 'ratio': e_ratio}}
  if mp: # multiprocessing is used, put in queue
    q.put(res)
  else: # multiprocessing is NOT used, update the dictionary
    q.update(res)
  print ("{} -> Loss: training = {}, test = {}, ratio = {}, epochs = {}".format(nn.name, e_tr, e_tst, e_ratio, total_epochs))

def get_nn_by_name(name, lnn):
  # get a neural network from a list of neural networks by name
  for nn in lnn:
    if nn.name == name:
      return nn
  return None

class Celosia:
  def __init__(self):
    self.rules = {}
    self.active_models = []

  def max_nh(self, ni, no, ns):
    '''a rule-of-thumb for the upper bound of the number of neurons in hidden layers:
       Ref: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
       where ni and no = number of input and outputs and,
       ns = number of samples in training data set.'''
    alpha = 2
    nh = math.ceil(ns/(alpha * (ni + no)))
    return nh

  def create_network_sigmoid(self, name, i, o, lh, eta=0.5):
    '''Create a neural network by making use of sigmoid activation function in all layers.
       Parameters:  name = name of the nn (any arbitrary string)
                    i,o =  number of neurons in [input|output] layers.
                    lh = list containing the number of neurons in each hidden layer.
                    eta = learning rate, 0.5 by default.'''
    assert type(i) is int, "parameter i (number of input neurons) needs to be an integer"
    assert type(o) is int, "parameter o (number of output neurons) needs to be an integer"
    assert type(lh) is list, "parameter h (list of neurons in each hidden layer) needs to be a list"
    assert len(lh) >= 1, "there needs to be at least one hidden layer"

    w = None # None means randomly initialize weights
    nn = Sequential(name, mse, eta)
    # input layer
    nn.add_layer(lh[0], sigmoid, 0.0, w, i)
    # hidden layers
    for h in lh[1:]:
      nn.add_layer(h, sigmoid, 0.0, w)
    # output layer
    nn.add_layer(o, sigmoid, 0.0, w)
    return nn

  def get_best_performing_nn(self, lnn, performance):
    '''Get the best performing NN.
       Parameters:  lnn = list of neural networks.
                    performance = dictionary of network performance.'''
    print ("Performance: {}".format(performance))
    opt_nn_name = min(performance, key=lambda k: performance[k]['vl']) # optimum neural network
    opt_nn = get_nn_by_name(opt_nn_name, lnn)
    print ("Winning NN: {}, error: {}".format(opt_nn_name, performance[opt_nn_name]['tl']))
    return opt_nn

  def create_optimal_network(self, inputs, outputs, config={}):
    '''create an optimal network by trying different structures.
       Parameters:  inputs, outputs = input and output vectors.
                    config = the configuration parameters (dict) as follows (empty by default):
                      N = number of different structures to try, default = 10.
                      epochs = number of epochs to try for each structure, default = 10000.
                      hmax = maximum number of hidden layers, default = 5.
                      nmax = maximum number of neurons in a hidden layer, default = 5.
                      view = view output (PDF), default = False.
                      mp = use mullti-processing, default = True'''
    # Load configuration
    start_time = time.time()
    N = config.get('N', 10)
    epochs = config.get('epochs', 10000)
    hmax = config.get('hmax', 5)
    nmax = config.get('nmax', 5)
    view = config.get('view', False)
    mp = config.get('mp', True)
    
    
    #nmax = max_nh(i, o, inputs.shape[0])
    lnn = [] # list of neural networks

    performance = {} # dictionary containing performance metrics for a NN
    if mp:
      q = Queue()
      lp = [] # list of processes

    i = inputs.shape[1] # number of colums in the input
    o = outputs.shape[1] # number of colums in the output
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.25, random_state=42)
    for j in range(N):
      lh = [] # list of the number of neurons in a hidden layer
      h = randint(1, hmax)
      for k in range(h):
        lh.append(randint(1, nmax))
      name = "structure-{}".format(j+1)
      nn = self.create_network_sigmoid(name, i, o, lh)
      lnn.append(nn)
      nn.draw(inputs, outputs, file="{}".format(nn.name), view=view, cleanup=True)
    for nn in lnn:
      if mp:
        p = Process(target=evaluate_nn, args=(nn, epochs, X_train, X_test, y_train, y_test, mp, q))
        lp.append(p)
        p.start()
      else:
        evaluate_nn(nn, epochs, X_train, X_test, y_train, y_test, mp, performance)
    if mp:
      for p in lp:
        p.join()
      while not q.empty():
        performance.update(q.get())
    opt_nn = self.get_best_performing_nn(lnn, performance)
    opt_nn.draw(inputs, outputs, file="most-optimal", view=view, cleanup=True)
    elapsed_time = time.time() - start_time
    print ("job completed in {} seconds.".format(elapsed_time))
