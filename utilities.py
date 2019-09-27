#!/usr/bin/Python
import pickle
from pylab import bone, pcolor, colorbar, plot, show

def save_object(obj, filename):
  with open(filename, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
  with open(filename, 'rb') as input:
    return pickle.load(input)

def plot_marker(xy, m, c):
  plot(xy[0] + 0.5,
       xy[1] + 0.5,
       m,
       markeredgecolor = c,
       markerfacecolor = 'None',
       markersize = 10,
       markeredgewidth = 2)