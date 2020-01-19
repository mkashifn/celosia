import numpy as np
from src.celosia import Celosia
from multiprocessing import freeze_support

#*************************************************************************
# An illustration example as presented in the following paper:
# Celosia: A Comprehensive Anomaly Detection Framework for Smart Cities
#*************************************************************************
def count_ones(n):
  # count the number of '1's in t a number's binary representation.
  return bin(n).count('1')

def to_bin_list(n, width):
  # convert a number of binary list with the specified width
  return [int(x) for x in '{:0{size}b}'.format(n,size=width)]

def create_xor_np_array(n):
  '''Crate an n-input XOR gate and returns inputs and outputs as numpy arrays.'''
  N = 2 ** n
  li = []
  lo = []
  for i in range(N):
    li.append(to_bin_list(i, n))
    o = 0
    if (count_ones(i) % 2):
      # number of '1's is odd
      o = 1
    lo.append([o])
  inputs = np.array(li)
  outputs = np.array(lo)
  return (inputs, outputs)


if __name__ == '__main__':
  freeze_support()
  (inputs, outputs) = create_xor_np_array(8)
  config = {
    'N':10, # number of different network structures to try
    #'view': True, # view the PDF file
  }
  celosia = Celosia()
  celosia.create_optimal_network('8-XOR',inputs, outputs, config)

  # https://github.com/alexarnimueller/som/blob/master/som.py