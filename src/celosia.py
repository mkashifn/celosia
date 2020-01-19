#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from progressive import Progressive
from random import randint
from utilities import save_object, load_object, plot_marker
import math
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process, Queue
from thirdparty.minisom import MiniSom
import numpy as np
import pandas as pd
from pylab import bone, pcolor, colorbar, plot, show

def evaluate_nn(nn, epochs, X_train, X_test, y_train, y_test, ed, imax, mp, q):
  e_ratio = 0
  suitable = False
  e_tst = 1000000
  total_epochs = 0
  while ((e_tst > ed) or (e_ratio < 1.0)) and (total_epochs < (imax * epochs)):
    total_epochs += epochs
    e_tr = nn.train(X_train, y_train, epochs, None, debug=True) # training loss
    e_tst = mse(y_test, nn.output(X_test)) # test loss
    e_ratio = e_tst / e_tr
  if (e_ratio >= 1.0) and (e_tst <= ed):
    suitable = True
  res = {nn.name:{'tl': e_tr, 'vl': e_tst, 'ratio': e_ratio, 'suitable': suitable, 'epochs': total_epochs, 's':nn.get_structure()}}
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

  def create_network_sigmoid(self, name, i, o, lh, eta):
    '''Create a neural network by making use of sigmoid activation function in all layers.
       Parameters:  name = name of the nn (any arbitrary string)
                    i,o =  number of neurons in [input|output] layers.
                    lh = list containing the number of neurons in each hidden layer.
                    eta = learning rate.'''
    assert type(i) is int, "parameter i (number of input neurons) needs to be an integer"
    assert type(o) is int, "parameter o (number of output neurons) needs to be an integer"
    assert type(lh) is list, "parameter h (list of neurons in each hidden layer) needs to be a list"
    assert len(lh) >= 1, "there needs to be at least one hidden layer"

    w = None # None means randomly initialize weights
    nn = Progressive(name, mse, eta)
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
    print ("Winning NN: {} : {}".format(opt_nn.name, p[opt_nn_name]))
    return opt_nn

  def create_optimal_network(self, name, inputs, outputs, config={}):
    '''create an optimal network by trying different structures.
       Parameters:  name = The name of the neural network
                    inputs, outputs = input and output vectors.
                    config = the configuration parameters (dict) as follows (empty by default):
                      N = number of different structures to try, default = 10.
                      epochs = number of epochs to try for each structure, default = 10000.
                      hmax = maximum number of hidden layers, default = 5.
                      nmax = maximum number of neurons in a hidden layer, default = 5.
                      ed = desired validation loss, default = 0.3.
                      imax = maximum number of iterations, default = 10.
                      eta = learning rate, default = 0.5.
                      view = view output (PDF), default = False.
                      mp = use mullti-processing, default = True.'''
    # Load configuration
    start_time = time.time()
    N = config.get('N', 10)
    epochs = config.get('epochs', 10000)
    hmax = config.get('hmax', 5)
    nmax = config.get('nmax', 5)
    ed = config.get('ed', 0.09)
    imax = config.get('imax', 10)
    eta = config.get('eta', 0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.25, random_state=randint(1, 125))

    for j in range(N):
      lh = [] # list of the number of neurons in a hidden layer
      h = randint(1, hmax)
      for k in range(h):
        lh.append(randint(1, nmax))
      ann_name = "{}-structure-{}".format(name, j+1)
      nn = self.create_network_sigmoid(ann_name, i, o, lh, eta)
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

    #calculate the accuracy of the opt_nn
    X = inputs
    Y = outputs
    Y_tmp = opt_nn.output(inputs)
    Y_pred = []
    for y_tmp in Y_tmp:
      y = (1 if y_tmp >= 0.50 else 0)
      Y_pred.append(y)
    (accuracy, fp, fn) = self.get_accuracy(Y, Y_pred)
    print ('Name={}, accuracy={}, false-positive={}, false-negative={}'.format(opt_nn.name, accuracy, fp, fn))
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
    map_y = map_x # square grid

    nb_features = X.shape[1] # number of features
    som = MiniSom(x = map_x, y = map_y, input_len = nb_features, sigma = 1.0, learning_rate = 0.5)
    som.random_weights_init(X)
    som.train_random(data = X, num_iteration = 2000)
    dm = som.distance_map()
    mid = []
    for x in X:
      w = som.winner(x)
      (x,y) = w
      mid.append(dm[x][y])

    self.dm = dm
    self.som = som
    self.grid = (map_x, map_y)
    return mid

  def plot_distance_map(self):
    '''Plots distance map. Need to call get_mid before calling this.'''
    bone()
    pcolor(self.dm.T)
    colorbar()
    show()

  def plot_distance_map_labels(self, X, Y):
    '''Plots distance map with labels. Need to call get_mid before calling this.
       Parameters: X = input features
                   Y = labels, 1 = normal, 0 = anomalous.'''
    red_set = set() # normal instances
    green_set = set() # anomalous instances
    for i, x in enumerate(X):
        w = self.som.winner(x)
        if int(Y[i]) == 0:
            red_set.add(w)
        else:
            green_set.add(w)
    bone()
    pcolor(self.dm.T)
    colorbar()
    (map_x, map_y) = self.grid
    for x in range(map_x):
      for y in range(map_y):
         xy = (x,y)
         if (xy in red_set) and (xy in green_set):
             plot_marker(xy, 'h', 'y')
         elif xy in red_set:
             plot_marker(xy, 'o', 'r')
         elif xy in green_set:
             plot_marker(xy, 's', 'g')
         else:
             pass #plot_marker(xy, 'v', 'b')
    show()

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
    fp = 0 # false positives (when prediction = normal, actual = anomalous)
    fn = 0 # false negatives (when prediction = anomalous, actual = normal)

    tp = 0 # total positives (normal)
    tn = 0 # total negatives (anomalous)

    for i in range(len(Y_pred)):
      correct += (1 if Y[i] == Y_pred[i] else 0)
      fp += (1 if ((Y[i] == 0) and (Y_pred[i] == 1)) else 0)
      fn += (1 if ((Y[i] == 1) and (Y_pred[i] == 0)) else 0)
      tp += (1 if Y[i] == 1 else 0)
      tn += (1 if Y[i] == 0 else 0)
    accuracy = (correct * 100) / total
    fp = (fp * 100) / tn
    fn = (fn * 100) / tp
    return (accuracy, fp, fn)

  def compute_threshold_vs_accuracy(self, mid, Y, step = 0.01):
    '''Computes the threshold vs accuracy, for all thresholds between 0 and 1.
       Parameters: mid = mean inter-neuron distance list obtained using get_mid()
                   Y = the expected or actual labels (1 = normal, 0 = anomalous)
                   step = the step to be taken from 0 to 1, default = 0.01'''
    th_v = [] # threshold vectors
    acc_v = [] # accuracy vector
    fp_v = [] # false positive vector
    fn_v = [] # false negative vector
    threshold = 0
    while threshold <= 1:
      Y_pred = self.label_data(mid, threshold)
      (accuracy, fp, fn) = self.get_accuracy(Y, Y_pred)
      th_v.append(threshold)
      acc_v.append(accuracy)
      fp_v.append(fp)
      fn_v.append(fn)
      threshold += step
    return (th_v, acc_v, fp_v, fn_v)

  def retrieve_labeled_data(self, source_list):
    '''Retrieve labeled data from different sources and scale the inputs in 0-1 range.
       Parameters: source_list = a list of data source dictionary with the following
                                 attributes:
                        filename = the name of the file
                        normal = 1 for normal 0 for anomalous label for this data
                        rows = the number of rows to read, -1 means all rows in the file.'''
    l_x = [] # list of features
    l_y = [] # list of labels
    for source in source_list:
      filename = source['filename']
      normal = bool(source['normal'])
      rows = int(source['rows'])
      ds = pd.read_csv(filename, index_col=0)
      if rows >= 0:
        x = ds.iloc[:rows, :].values
      else: # fetch all rows
        x = ds.iloc[:, :].values

      if normal: # 1 represents normal
        y = np.ones(x.shape[0])
      else:
        y = np.zeros(x.shape[0])

      l_x.append(x)
      l_y.append(y)
    X = np.concatenate(tuple(l_x), axis=0)
    Y = np.concatenate(tuple(l_y), axis=0)

    # scale the input in 0-1 range
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    return (X, Y)
