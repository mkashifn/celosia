import numpy as np
from celosia import Celosia
from multiprocessing import freeze_support
import pandas as pd
import os
import matplotlib.pyplot as plt

#3050, 250
#20000, 100
#19800, 200

Danmini_Doorbell_benign = {'filename': 'data/Danmini_Doorbell_benign_traffic.csv', 'normal':True, 'rows':1980}
Danmini_Doorbell_gafgyt_scan = {'filename': 'data/Danmini_Doorbell_gafgyt_scan.csv', 'normal':False, 'rows':20}

source_list = [Danmini_Doorbell_benign, Danmini_Doorbell_gafgyt_scan]
#*************************************************************************
# An illustration example for automatic data labelling
#*************************************************************************

def main(source_list):
  celosia = Celosia()
  (X, Y) = celosia.retrieve_labeled_data(source_list)
  mid = celosia.get_mid(X)
  Y_pred = celosia.label_data(mid, 0.24)
  (accuracy, fp, fn) = celosia.get_accuracy(Y, Y_pred)
  print ('accuracy = {}, fp = {}, fn = {}'.format(accuracy, fp, fn))
  (th_v, acc_v, fp_v, fn_v) = celosia.compute_threshold_vs_accuracy(mid, Y)
  #celosia.plot_distance_map_labels(X, Y)

  plt.subplot(3, 1, 1)
  plt.plot(th_v, acc_v)
  plt.ylabel('accuracy')
  plt.xlabel('threshold')

  plt.subplot(3, 1, 2)
  plt.plot(th_v, fp_v, 'g')
  plt.ylabel('false positives')
  plt.xlabel('threshold')

  plt.subplot(3, 1, 3)
  plt.plot(th_v, fn_v, 'r')
  plt.ylabel('false negatives')
  plt.xlabel('threshold')

  plt.show()

if __name__ == '__main__':
  freeze_support()
  main(source_list)