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

def get_data(device, count_normal, count_anomalous):
  dataset1 = pd.read_csv('evaluation/{}_benign_traffic.csv'.format(device))
  dataset2 = pd.read_csv('evaluation/{}_attack_combo.csv'.format(device))

  X1 = dataset1.iloc[:count_normal, :].values
  y1 = np.ones((X1.shape[0],1))
  X2 = dataset2.iloc[:count_anomalous, :].values
  y2 = np.zeros((X2.shape[0],1))

  X = np.concatenate((X1, X2), axis=0)
  Y = np.concatenate((y1, y2), axis=0)
  X = np.array(X)
  Y = np.array(Y)
  sc = MinMaxScaler(feature_range = (0, 1))
  X = sc.fit_transform(X)
  return (X, Y)