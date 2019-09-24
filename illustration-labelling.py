import numpy as np
from celosia import Celosia
from multiprocessing import freeze_support
import pandas as pd

#*************************************************************************
# An illustration example for automatic data labelling
#*************************************************************************
def main():
  ds = pd.read_csv('data/SampleData.csv')
  X = ds.iloc[:, :-1].values
  Y = ds.iloc[:, -1].values
  celosia = Celosia()
  Y_pred = celosia.label_data(X)
  print len(Y_pred), len(Y)
  '''for i in range(len(Y)):
    print Y[i], Y_pred[i]'''

if __name__ == '__main__':
  freeze_support()
  main()
