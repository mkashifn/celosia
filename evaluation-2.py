import numpy as np
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object
import math
import pandas as pd
from multiprocessing import freeze_support
from celosia import Celosia

#19800, 200

'''
def get_error(inputs, outputs):
  i = inputs.shape[1] # number of colums in the input
  o = outputs.shape[1] # number of colums in the output
  w = None # None means randomly initialize weights
  nn = Sequential(mse, 0.5)
  # input layer
  nn.add_layer(4, sigmoid, 0.0, w, i)
  # hidden layers
  nn.add_layer(6, sigmoid, 0.0, w)
  nn.add_layer(3, sigmoid, 0.0, w)
  # output layer
  nn.add_layer(o, sigmoid, 0.0, w)
  epochs = [2000, 2000, 2000, 2000, 2000]
  err = []
  for ep in epochs:
    e = nn.train(inputs, outputs, ep)
    err.append(e)
  return err
'''

def get_data(device):
  dataset1 = pd.read_csv('evaluation/{}_benign_traffic.csv'.format(device))
  dataset2 = pd.read_csv('evaluation/{}_attack_combo.csv'.format(device))

  X1 = dataset1.iloc[:19800, :].values
  y1 = np.ones((X1.shape[0],1))
  X2 = dataset2.iloc[:200, :].values
  y2 = np.zeros((X2.shape[0],1))

  X = np.concatenate((X1, X2), axis=0)
  Y = np.concatenate((y1, y2), axis=0)
  X = np.array(X)
  Y = np.array(Y)
  return (X, Y)

def evaluate(id, device):
  (X, Y) = get_data(device)
  celosia = Celosia()
  mid = celosia.get_mid(X)
  Y_pred = celosia.label_data(mid, 0.24)
  (accuracy, fp, fn) = celosia.get_accuracy(Y, Y_pred)
  print ('ID={}, accuracy={}, false-positive={}, false-negative={}'.format(id, accuracy, fp, fn))

devices = [('Danmini', 'Danmini_Doorbell'),
           ('Ecobee', 'Ecobee_Thermostat'),
           ('Ennio', 'Ennio_Doorbell'),
           ('Philips B120N10', 'Philips_B120N10_Baby_Monitor'),
           ('Provision PT737E', 'Provision_PT_737E_Security_Camera'),
           ('Provision PT838', 'Provision_PT_838_Security_Camera'),
           ('Samsung SNH1011', 'Samsung_SNH_1011_N_Webcam'),
           ('SimpleHome XCS71002', 'SimpleHome_XCS7_1002_WHT_Security_Camera'),
           ('SimpleHome XCS71003', 'SimpleHome_XCS7_1003_WHT_Security_Camera'),
          ]

def main():
  for device in devices:
    evaluate(device[0], device[1])

if __name__ == '__main__':
  freeze_support()
  main()