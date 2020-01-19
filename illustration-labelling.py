import numpy as np
from src.celosia import Celosia
from multiprocessing import freeze_support
import pandas as pd
import os

#*************************************************************************
# An illustration example for automatic data labelling
#*************************************************************************
def main(filename):
  path, name = os.path.split(filename)
  name, ext = os.path.splitext(name)
  output_filename = os.path.join(path, name + '-Labeled' + ext)

  ds = pd.read_csv(filename,
                  index_col=0)
  X = ds.iloc[:, :-1].values
  Y = ds.iloc[:, -1].values
  celosia = Celosia()
  mid = celosia.get_mid(X)
  Y_pred = celosia.label_data(mid, 0.02)
  accuracy = celosia.get_accuracy(Y, Y_pred)
  print ('accuracy = ', accuracy)
  ds['SOM-mid'] = mid
  ds['Y_pred_0.02'] = Y_pred
  ds.to_csv(output_filename)
  (th_v, acc_v, fp_v, fn_v) = celosia.compute_threshold_vs_accuracy(mid, Y)
  print th_v
  print acc_v

if __name__ == '__main__':
  freeze_support()
  main('data/Danmini_Doorbell.csv')