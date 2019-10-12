#!/usr/bin/Python
import pickle
from pylab import bone, pcolor, colorbar, plot, show
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def save_object(obj, filename):
  with open(filename, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
  with open(filename, 'rb') as input:
    return pickle.load(input)

def plot_marker(xy, m, c):
  plot(xy[0] + 0.5,
       xy[1] + 0.5,
       m,
       markeredgecolor = c,
       markerfacecolor = 'None',
       markersize = 10,
       markeredgewidth = 2)

def retrieve_labeled_data(source_list, anomaly_label=0):
  '''Retrieve labeled data from different sources and scale the inputs in 0-1 range.
     Parameters: source_list = a list of data source dictionary with the following
                               attributes:
                      filename = the name of the file
                      normal = label 1 for normal and 'anomaly_label' for anomalous entries
                      rows = the number of rows to read, -1 means all rows in the file.
                 anomaly_label = the value to label anomalies, default = 0'''
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

    y = np.ones(x.shape[0]) # 1 represents normal
    if not normal: # not normal, label as anomalous
      y = anomaly_label * y

    l_x.append(x)
    l_y.append(y)
  X = np.concatenate(tuple(l_x), axis=0)
  Y = np.concatenate(tuple(l_y), axis=0)
  X = np.array(X)
  Y = np.array(Y)
  Y = np.reshape(Y, (-1, 1))
  # scale the input in 0-1 range
  sc = MinMaxScaler(feature_range = (0, 1))
  X = sc.fit_transform(X)
  #print ('Shapes = ', X.shape, Y.shape)
  return (X, Y)

def get_device_data(device, count_normal, count_anomalous, anomaly_label=0):
  '''Get device data from evaluation directory
     Parameters: device = device name
                 count_normal = the number of entries to retrieve from normal record
                 count_anomalous = the number of entries to retrieve from anomalous record
                 anomaly_label = the value to label anomalies, default = 0'''
  file_normal = 'evaluation/{}_benign_traffic.csv'.format(device)
  file_anomalous = 'evaluation/{}_attack_combo.csv'.format(device)

  normal = {'filename':file_normal, 'normal':True, 'rows':count_normal}
  anomalous = {'filename':file_anomalous, 'normal':False, 'rows':count_anomalous}

  source_list = [normal, anomalous]

  return retrieve_labeled_data(source_list, anomaly_label)

def get_accuracy(Y, Y_pred):
  '''Return accuracy of the prediction on a scale of 0-1.
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
  offset = 1e-7 # add offset to avoid divide-by-zero exception
  total = float(total) + offset
  tn = float(tn) + offset
  tp = float(tp) + offset
  accuracy = float(correct)/ total
  fp = float(fp) / tn
  fn = float(fn) / tp
  return (accuracy, fp, fn)

#19800, 200

# *****************************************************************
# Evaluate Performance of Different Techniques
# *****************************************************************

def scale_output_0_1(Y_real):
  Y_pred = []
  for y_real in Y_real:
    y = (1 if y_real >= 0.5 else 0)
    Y_pred.append(y)
  return Y_pred