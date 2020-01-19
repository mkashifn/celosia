import numpy as np
from src.functions import sigmoid
from src.estimators import mse
from src.progressive import Progressive
from random import randint
from utilities import save_object, load_object, get_device_data
import math
import pandas as pd
from multiprocessing import freeze_support
from src.celosia import Celosia

#19800, 200

# *****************************************************************
# Evaluate Neural Network Generation and Performance of Winning NN
# *****************************************************************

def evaluate(id, device):
  (X, Y) = get_device_data(device, 19800, 200)

  print ("{}: {}, {}".format(device, X.shape, Y.shape))

  config = {
    #'N':10, # number of different network structures to try
    #'view': True, # view the PDF file
    'hmax': 5,
    'nmax': 5,
    'N': 1,
    'epochs': 10000,
    'eta': 0.75,
    'imax': 1,
  }
  celosia = Celosia()
  celosia.create_optimal_network(device,X, Y, config)

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
  freeze_support()
  main()