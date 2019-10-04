import numpy as np
from functions import sigmoid
from estimators import mse
from sequential import Sequential
from random import randint
from utilities import save_object, load_object, get_data
import math
import pandas as pd
from multiprocessing import freeze_support
from celosia import Celosia

#19800, 200

# *****************************************************************
# Evaluate the performance of Self-Organizing Maps (SOM)
# *****************************************************************

def evaluate(id, device):
  (X, Y) = get_data(device, 19800, 200)
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