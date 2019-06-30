import numpy as np
import celosia

#*************************************************************************
# An illustration example as presented in the following paper:
# Celosia: A Comprehensive Anomaly Detection Framework for Smart Cities
#*************************************************************************

# A 3-input XOR Gate
inputs = np.array([
                  [0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1],
                  ])
outputs = np.array([
                   [0],
                   [1],
                   [1],
                   [0],
                   [1],
                   [0],
                   [0],
                   [1],
                   ])

i = inputs.shape[1] # number of colums in the input
o = outputs.shape[1] # number of colums in the output
n = 10 # number of different network structures to try
celosia.create_optimal_network(i, o, n, inputs, outputs)