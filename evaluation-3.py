import numpy as np
#from functions import sigmoid, softmax
#from estimators import mse
#from sequential import Sequential
#from random import randint
from utilities import get_device_data
#import math
import pandas as pd
#from multiprocessing import freeze_support
#from celosia import Celosia
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

def get_accuracy(Y, Y_pred):
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

#19800, 200

# *****************************************************************
# Evaluate Performance of Different Techniques
# *****************************************************************

'''
def get_error(name, inputs, outputs):
  i = inputs.shape[1] # number of colums in the input
  o = outputs.shape[1] # number of colums in the output
  w = None # None means randomly initialize weights
  nn = Sequential(name, mse, 0.5)
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
    e = nn.train(inputs, outputs, ep, debug=True)
    err.append(e)
  return err
'''
def scale_output_0_1(Y_real):
  Y_pred = []
  for y_real in Y_real:
    y = (1 if y_real >= 0.5 else 0)
    Y_pred.append(y)
  return Y_pred

def get_error(name, inputs, outputs):
  i = inputs.shape[1] # number of columns in the input
  o = outputs.shape[1] # number of columns in the output
  # Initialising the ANN
  classifier = Sequential()

  # Adding the input layer and the first hidden layer
  classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = i))

  # Adding the second hidden layer
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
  
  # Adding the third hidden layer
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

  # Adding the output layer
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

  # Compiling the ANN
  classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

  # Fitting the ANN to the Training set
  classifier.fit(inputs, outputs, batch_size = 10, epochs = 20)

  # Part 3 - Making predictions and evaluating the model

  # Predicting the Test set results
  y_pred = classifier.predict(inputs)
  Y_pred = scale_output_0_1(y_pred)
  Y = outputs
  (accuracy, fp, fn) = get_accuracy(Y, Y_pred)
  print ('name={}, accuracy={}, false-positive={}, false-negative={}'.format(name, accuracy, fp, fn))
  return 10

def get_data(device, count_normal, count_anomalous, anomaly_label=0):
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

def evaluate(name, device):
  (X, Y) = get_device_data(device, 1980, 1980, anomaly_label=0)
  #(X, Y) = get_data(device, 1980, 1980, anomaly_label=0)

  #celosia = Celosia()
  #mid = celosia.get_mid(X)
  #Y_pred = celosia.label_data(mid, 0.24)
  #(accuracy, fp, fn) = celosia.get_accuracy(Y, Y_pred)
  #print ('name={}, accuracy={}, false-positive={}, false-negative={}'.format(name, accuracy, fp, fn))
  err = get_error(name, X, Y)
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