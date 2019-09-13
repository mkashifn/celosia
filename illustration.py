import numpy as np
from celosia import Celosia

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

N = 10 # number of different network structures to try
celosia = Celosia()
celosia.create_optimal_network(inputs, outputs, N)