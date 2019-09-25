#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object
import math
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process, Queue
from thirdparty.minisom import MiniSom

def evaluate_nn(nn, epochs, X_train, X_test, y_train, y_test, ed, imax, mp, q):
  e_ratio = 0
  suitable = False
  e_tst = 1000000
  total_epochs = 0
  while ((e_tst > ed) or (e_ratio < 1.0)) and (total_epochs < (imax * epochs)):
    total_epochs += epochs
    e_tr = nn.train(X_train, y_train, epochs) # training loss
    e_tst = mse(y_test, nn.output(X_test)) # test loss
    e_ratio = e_tst / e_tr
  if (e_ratio >= 1.0) and (e_tst <= ed):
    suitable = True
  res = {nn.name:{'tl': e_tr, 'vl': e_tst, 'ratio': e_ratio, 'suitable': suitable}}
  if mp: # multiprocessing is used, put in queue
    q.put(res)
  else: # multiprocessing is NOT used, update the dictionary
    q.update(res)
  #print ("{} -> Loss: training = {}, test = {}, ratio = {}, epochs = {}, suitable = {}".format(nn.name, e_tr, e_tst, e_ratio, total_epochs, suitable))
  print (res)

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
    #print ("Performance: {}".format(performance))
    p = performance
    opt_nn_name = min(p, key=lambda k: p[k]['vl'] if p[k]['suitable'] else 1000000) # optimum neural network
    opt_nn = get_nn_by_name(opt_nn_name, lnn)
    print ("Winning NN: {}, tl: {}, vl: {}".format(opt_nn_name, p[opt_nn_name]['tl'], p[opt_nn_name]['vl']))
    return opt_nn

  def create_optimal_network(self, inputs, outputs, config={}):
    '''create an optimal network by trying different structures.
       Parameters:  inputs, outputs = input and output vectors.
                    config = the configuration parameters (dict) as follows (empty by default):
                      N = number of different structures to try, default = 10.
                      epochs = number of epochs to try for each structure, default = 10000.
                      hmax = maximum number of hidden layers, default = 5.
                      nmax = maximum number of neurons in a hidden layer, default = 5.
                      ed = desired validation loss, default = 0.3.
                      imax = maximum number of iterations, default = 10.
                      view = view output (PDF), default = False.
                      mp = use mullti-processing, default = True.'''
    # Load configuration
    start_time = time.time()
    N = config.get('N', 10)
    epochs = config.get('epochs', 10000)
    hmax = config.get('hmax', 5)
    nmax = config.get('nmax', 5)
    ed = config.get('ed', 0.3)
    imax = config.get('imax', 10)
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
        p = Process(target=evaluate_nn, args=(nn, epochs, X_train, X_test, y_train, y_test, ed, imax, mp, q))
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

  def get_mid(self, data, map_x = 20):
    '''Get mean inter-neuron distance using SOM.
       Parameters: data = a numpy array and it should contain
                        all columns as features and any manually
                        labeled columns should be removed before
                        calling this function.
                    map_x = square-grid size, default = 20'''
    X = data
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)

    map_y = map_x # square grid

    nb_features = X.shape[1] # number of features
    som = MiniSom(x = map_x, y = map_y, input_len = nb_features, sigma = 1.0, learning_rate = 0.5)
    som.random_weights_init(X)
    som.train_random(data = X, num_iteration = 100)
    dm = som.distance_map()
    mid = []
    for i, x in enumerate(X):
      w = som.winner(x)
      (x,y) = w
      mid.append(dm[x][y])
    return mid

  def label_data(self, mid, threshold = 0.02):
    '''Label data (predict as normal or anomalous) based upon mean inter-neuron distance.
       Parameters: mid = mean inter-neuron distance list obtained using get_mid()
                   threshold = the threshold (default = 0.02) that is used to
                               determine if normal = 1 (when mid <= threshold),
                               or anomalous = 0 otherwise.'''
    Y_pred = []
    for m in mid:
      normal = (1 if m <= threshold else 0)
      Y_pred.append(normal)
    return Y_pred

  def get_accuracy(self, Y, Y_pred):
    '''Return accuracy of the prediction as a percentage.
       Parameters: Y = the expected or actual labels (1 = normal, 0 = anomalous)
                   Y_pred = the predicted output obtained using label_data().'''
    #assert len(Y) == len (Y_pred), "Y and Y_pred are of different dimensions"
    total = len(Y_pred)
    correct = 0
    for i in range(len(Y_pred)):
      correct += (1 if Y[i] == Y_pred[i] else 0)
    return (correct * 100) / total
