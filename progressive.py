#!/bin/python
import numpy as np
from graphviz import Digraph
from graphviz import Graph
import warnings
from utilities import get_accuracy, scale_output_0_1
import time

class Layer:
  def __init__(self, activation, bias, weights):
    self.a = activation
    self.b = bias
    self.w = weights
    self.old_vt = 0 * self.w # old vt (weight difference)
    self.count = weights.shape[1] #number of neurons
    self.sigma = None
    self.lto = None # last training outputs
  def output(self, input):
    self.i = input
    self.o = self.a.fx(np.dot(self.i, self.w) + self.b)
    return self.o

class Progressive:
  def __init__(self, name, loss, eta):
    self.name = name
    self.layers = []
    self.loss = loss
    self.eta = eta # learning rate
    self.input_layer_size = None #number of neurons in the input layer
    self.output_layer_size = None #number of neurons in the output layer

    self.layers = []
    self.layer_count = 0
    self.weight_count = 1

    self.retraining_required = False

  def get_structure(self):
    s = []
    for l in self.layers:
      s.append(l.count)
    return s

  def get_weights(self):
    w = {}
    i = 1
    for l in self.layers:
      w.update({i:l.w})
      i += 1
    return w

  def weights_initializer(self, initial_weights, required_shape):
    weights = initial_weights
    if weights is None:
      x = required_shape[0]
      y = required_shape[1]
      weights = np.random.rand(x,y)
    if weights.shape != required_shape:
      warnings.warn("wights matrix is not properly formed, expected {e} vs actual {a}".format(e=required_shape, a=initial_weights.shape), UserWarning)
      weights.shape = required_shape
    return weights

  def add_layer(self, neuron_count, activation, bias, initial_weights=None, input_size=None):
    # Validate the inputs
    if self.output_layer_size is None and input_size is None:
      raise ValueError('input_size needs to be defined for the very first layer')
    if self.output_layer_size is not None and input_size is not None:
      raise ValueError('input_size is not required for this layer')
    if input_size is not None:
      layer_input_size = input_size
      self.input_layer_size = input_size
    else:
      layer_input_size = self.output_layer_size # current input is the output of last layer
    expected_weights_shape = (layer_input_size, neuron_count)
    weights = self.weights_initializer(initial_weights, expected_weights_shape)
    self.layers.append(Layer(activation, bias, weights))
    self.output_layer_size = neuron_count

  def add_connections_to_the_left(self, index, additional_neron_count, initial_weights):
    if index >= len(self.layers):
      raise ValueError('invalid layer index')
    layer = self.layers[index]
    shape = layer.w.shape
    new_shape = (additional_neron_count, shape[1])
    weights = self.weights_initializer(initial_weights, new_shape)
    new_weights = np.append(weights, layer.w, axis = 0)
    self.input_layer_size += additional_neron_count
    layer.w = new_weights
    self.retraining_required = True

  def add_connections_to_the_right(self, index, additional_neron_count, initial_weights):
    if index >= len(self.layers) - 1: # can not expand output layer
      raise ValueError('invalid layer index')
    layer = self.layers[index]
    shape = layer.w.shape
    new_shape = (shape[0], additional_neron_count)
    weights = self.weights_initializer(initial_weights, new_shape)
    new_weights = np.append(layer.w, weights, axis = 1)
    self.output_layer_size += additional_neron_count
    layer.count += additional_neron_count
    layer.w = new_weights
    self.retraining_required = True

  def expand_input_layer(self, additional_neron_count, initial_weights=None):
    if additional_neron_count <= 0 or len(self.layers) <= 0:
      return 0
    self.add_connections_to_the_left(0, additional_neron_count, initial_weights)


  def expand_layer(self, index, additional_neron_count, initial_weights=None):
    if additional_neron_count <= 0 or len(self.layers) <= 0:
      return 0
    self.add_connections_to_the_right(index, additional_neron_count, initial_weights)
    self.add_connections_to_the_left(index+1, additional_neron_count, initial_weights)

  def retrain_layer_1(self, inputs, epochs):
    if self.retraining_required == False:
      return
    layer = self.layers[0]
    targets = layer.lto
    for i in range(epochs):
      layer.output(inputs)
      sigma = -self.loss.dfx(targets,layer.o)*layer.a.dfx(layer.o)
      nw = self.update_layer_weights(layer, sigma, self.eta)
    self.retraining_required = False

  def retrain_layer(self, index, inputs, epochs):
    if self.retraining_required == False:
      return
    if index >= len(self.layers) - 1: # can not expand output layer
      raise ValueError('invalid layer index')
    targets = self.layers[index + 1].lto
    for i in range(epochs):
      layer = self.layers[index + 1]
      self.feed_forward(inputs)
      sigma = -self.loss.dfx(targets,layer.o)*layer.a.dfx(layer.o)
      old_weights = self.update_layer_weights(layer, sigma, self.eta)
      layer = self.layers[index]
      sigma = np.dot(sigma, old_weights.T) * layer.a.dfx(layer.o)
      self.update_layer_weights(layer, sigma, self.eta)
    self.retraining_required = False

  def feed_forward(self, input):
    if self.retraining_required == True:
      warnings.warn("Retrain the network using retrain_layer_1() first.", UserWarning)
    output = None
    for l in self.layers:
      output = l.output(input)
      input = output
    self.o = output
    return self.o

  def output(self, input):
    return self.feed_forward(input)

  def train(self, inputs, targets, epochs, batch_size, debug=False):
    def finish_training(layers):
      for l in layers:
        l.lto = l.o

    A = targets
    B = self.feed_forward(inputs)
    if debug:
        print ("Before Training: Loss = {loss}".format(loss = self.loss(A, B)))
    M = inputs.shape[0] # number of rows in the input
    n_batches = int(np.ceil(float(M)/float(batch_size)))
    # stochastic gradient descent, 
    inputs_r = inputs
    targets_r = targets
    for i in range(epochs):
      print ("Epoch: {}/{}".format(i+1, epochs))
      rand_indices = np.random.permutation(M)
      inputs_r = inputs_r[rand_indices] #.reshape(inputs.shape[0], inputs.shape[1])
      targets_r = targets_r[rand_indices] #.reshape(targets.shape[0], targets.shape[1])
      for b in range(n_batches):
        start_index = b * batch_size
        end_index = start_index + batch_size
        if end_index > M:
          end_index = M
        inputs_b = inputs_r[start_index:end_index, :]
        targets_b = targets_r[start_index:end_index, :]
        B = self.feed_forward(inputs_b)
        self.propagate_back(targets_b)

        B = self.feed_forward(inputs_b)
        Y_pred = scale_output_0_1(B)
        Y = targets_b
        (accuracy, fp, fn) = get_accuracy(Y, Y_pred)
        print ("    {e}/{m}, Loss = {loss}, Accuracy={accuracy}".format(e=end_index, m=M, loss = self.loss(inputs_b, B), accuracy=accuracy))
        
        #print ("Weights = {}".format(self.get_weights()))
        #time.sleep(2)

      '''B = self.feed_forward(inputs)
      Y_pred = scale_output_0_1(B)
      Y = targets
      (accuracy, fp, fn) = get_accuracy(Y, Y_pred)
      if debug and (i%100) >= 0:
        print ("Epoch: {i}, Loss = {loss}, Accuracy={accuracy}".format(i=i, loss = self.loss(A, B), accuracy=accuracy))'''
    B = self.feed_forward(inputs)
    finish_training(self.layers)
    return self.loss(A, B)

  def update_layer_weights(self, layer, sigma, eta, momentum=0):
    vt = (momentum * layer.old_vt) + np.dot(layer.i.T, sigma) * eta
    layer.old_vt = vt
    new_weights = layer.w - vt
    old_weights = layer.w
    layer.w = new_weights
    layer.sigma = sigma
    return old_weights

  def propagate_back(self, targets):
    layers = self.layers[::-1] # reverse, need to start from output layer
    layer = layers[0]
    #print(self.loss.dfx(targets,layer.o))
    sigma = -self.loss.dfx(targets,layer.o)*layer.a.dfx(layer.o)
    old_weights = self.update_layer_weights(layer, sigma, self.eta)
    layers = layers[1:] # other layers
    for layer in layers:
      sigma = np.dot(sigma, old_weights.T) * layer.a.dfx(layer.o)
      old_weights = self.update_layer_weights(layer, sigma, self.eta)

  def draw(self, inputs, targets, file="sequential", dir="draw", view=True, cleanup=False):
    graph = Digraph(directory='graphs', format='pdf',
                  graph_attr=dict(ranksep='2', rankdir='LR', color='white', splines='line'),
                  node_attr=dict(label='', shape='circle', width='0.1'))

    np_formatter = {'float_kind':lambda x: "%.9g" % x}

    def increment_glc():
      self.layer_count += 1

    def increment_gwc():
      self.weight_count += 1

    def reset_gxc():
      self.layer_count = 0
      self.weight_count = 1

    reset_gxc()

    def draw_cluster(name, length, values, fillcolor="#FFFFFF", subscript="", targets=None):
      names = []
      with graph.subgraph(name='cluster_{name}'.format(name=name)) as c:
        c.attr(label='{name}\n(layer {glc})'.format(name=name, glc = self.layer_count))
        increment_glc()

        for i in range(length):
          name_str = '{name}_{i}'.format(name=name, i=i)
          outcome = values[:,i]
          label = '{id}{ss}-{i}\n= {val}'.format(ss=subscript,id=name[0],i=i, val='{}'.format(np.array2string(outcome, formatter=np_formatter)))
          if targets is not None:
            target = targets[:,i]
            loss = self.loss(target, outcome) / targets.shape[1]
            label += '\ntarget: {}'.format(np.array2string(target, formatter=np_formatter))
            label += '\nloss: {}'.format(np.array2string(loss, formatter=np_formatter))
          #label = "\$\sigma_1\$"
          #label = u'\u0220' #Unicode: https://unicode-table.com/en/#control-character
          c.node(name_str, label, color='black',style='filled',fillcolor=fillcolor)
          #c.node(name_str, r"$\sigma_1$")
          names.append(name_str)
      return names

    def draw_connections(src, dst, dst_layer):
      auto_colors = ['#00A2EB','#4BC000','#9966FF','#FF7700','#808000','#800000','#0020F0','#00F0F0','#8463FF','#E6194B']
      w = dst_layer.w.flatten('F')

      i = 0;
      for d in dst:
        for s in src:
          color = auto_colors[i % len(auto_colors)]
          graph.edge(s, d, fontcolor=color, color=color, label='w{wl} = {weight}'.format(wl=self.weight_count, weight=w[i]))
          increment_gwc()
          i += 1

    A = targets
    B = self.feed_forward(inputs)

    src = draw_cluster('input', inputs.shape[1], inputs, "#FFFF00")
    i = 1
    layer_input = inputs
    for layer in self.layers[:-1]:
      layer_output = layer.output(layer_input)
      dst = draw_cluster('hidden_layer_{i}'.format(i=i), layer.count, layer_output, "#04ADFC", i)
      i += 1
      draw_connections(src, dst, layer)
      src = dst
      layer_input = layer_output

    layer = self.layers[-1]
    dst = draw_cluster('output', layer.count, layer.output(layer_input), "#00FF00", targets=targets)

    draw_connections(src, dst, layer)

    tl = self.loss(A, B)
    tl = np.array2string(tl, formatter=np_formatter)
    graph.attr(label='{file}, total loss = {tl}'.format(file=file, tl=tl))

    graph.render(file, dir, view=view, cleanup=cleanup)