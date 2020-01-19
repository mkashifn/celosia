import numpy as np
from src.functions import sigmoid, softmax, relu
from src.estimators import mse, cross_entropy
from src.optimizers import adam_default, momentum_default
from src.progressive import Progressive
from random import randint
from utilities import get_device_data, scale_output_0_1, get_accuracy
import pandas as pd
#from multiprocessing import freeze_support
#from celosia import Celosia
from sklearn.preprocessing import MinMaxScaler

def get_error_progressive(name, inputs, outputs):
  i = inputs.shape[1] # number of colums in the input
  o = outputs.shape[1] # number of colums in the output
  w = None # None means randomly initialize weights
  nn = Progressive(name, mse, 1, None)
  # input layer
  nn.add_layer(4, relu, 0.0, w, i)
  # hidden layers
  nn.add_layer(6, relu, 0.0, w)
  nn.add_layer(6, relu, 0.0, w)
  # output layer
  nn.add_layer(o, sigmoid, 0.0, w)
  #epochs = [2000, 2000, 2000, 2000, 2000]
  epochs = [100]
  err = []
  for ep in epochs:
    e = nn.train(inputs, outputs, ep, 4000, debug=True)
    err.append(e)
  y_pred = nn.output(inputs)
  Y_pred = scale_output_0_1(y_pred)
  Y = outputs
  (accuracy, fp, fn) = get_accuracy(Y, Y_pred)
  print ('name={}, accuracy={}, false-positive={}, false-negative={}'.format(name, accuracy, fp, fn))
  return err


def evaluate(name, device):
  (X, Y) = get_device_data(device, 2000, 2000, anomaly_label=0)
  #(X, Y) = get_data(device, 1980, 1980, anomaly_label=0)

  #celosia = Celosia()
  #mid = celosia.get_mid(X)
  #Y_pred = celosia.label_data(mid, 0.24)
  #(accuracy, fp, fn) = celosia.get_accuracy(Y, Y_pred)
  #print ('name={}, accuracy={}, false-positive={}, false-negative={}'.format(name, accuracy, fp, fn))
  err = get_error_progressive(name, X, Y)
  print (err)

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
devices = [('Danmini', 'Danmini_Doorbell')]
def main():
  for device in devices:
    evaluate(device[0], device[1])

if __name__ == '__main__':
  #freeze_support()
  main()