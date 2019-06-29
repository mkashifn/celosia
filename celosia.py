#!/usr/bin/Python
from functions import sigmoid
from estimators import mse
from sequential import Sequential
import numpy as np
from utilities import save_object, load_object

inputs = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
outputs = np.array([[0],[1],[1],[0]])
weights = None
nn = Sequential(mse, 1)
nn.add_layer(4, sigmoid, 0.0, weights, 3)
nn.add_layer(3, sigmoid, 0.0, weights)
nn.add_layer(1, sigmoid, 0.0, weights)
#nn.expand_input_layer(2)
#exit(0)
epochs = 1500
nn.train(inputs, outputs, epochs)
print (nn.output(inputs[0,:]))
print (nn.output(inputs[1,:]))
print (nn.output(inputs[2,:]))
print (nn.output(inputs[3,:]))
#nn.draw(np.array([inputs[0, :]]), np.array([outputs[0,:]]))
nn.draw(inputs, outputs, file="tmp", cleanup=True)
print "NN: ", nn.output(inputs)
filename = 'trained-nn.nn'
save_object(nn, filename)
del nn
nnl = load_object(filename)
print "NN1: ", nnl.output(inputs)
nnl.expand_input_layer(2)
inputs = np.array([[2,3,0,0,1],
                  [4,5,0,1,1],
                  [6,7,1,0,1],
                  [8,9,1,1,1]])
nnl.draw(inputs, outputs, file="loaded-nn", cleanup=True)
nnl.retrain_layer_1(inputs, epochs)
nnl.draw(inputs, outputs, file="retrained-L0", cleanup=True)

nnl.expand_layer(1, 3)
nnl.draw(inputs, outputs, file="expanded-L1", cleanup=True)
nnl.retrain_layer(1, inputs, epochs)
nnl.draw(inputs, outputs, file="retrained-L1", cleanup=True)