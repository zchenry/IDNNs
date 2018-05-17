import functools
import tensorflow as tf
import numpy as np
from idnns.networks.models import multi_layer_perceptron
from idnns.networks.models import deepnn
from idnns.networks.ops import *

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        # print hasattr(self, attribute)
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model:
    def __init__(self, input_size, layers, num_of_classes, lr_local=0.001,
                     save_file='', act=0, cov=False):
        self.cov = cov
        self.input_size = input_size
        self.layers = layers
        self.all_layers = np.copy(layers)
        self.all_layers = np.insert(self.all_layers, 0, input_size)
        self.num_of_classes = num_of_classes
        self._num_of_layers = len(layers) + 1
        self.lr_local = lr_local
        self._save_file = save_file
        self.hidden = None
        self.savers = []
        self.act = tf.nn.relu if act == 1 else tf.nn.tanh

    def initilizae_layer(self, name_scope, row_size, col_size, act, last_hidden):
        weights = get_scope_variable(
            name_scope=name_scope, var="weights", shape=[row_size, col_size],
            initializer=tf.truncated_normal_initializer(
                mean=0.0, stddev=1.0 / np.sqrt(float(row_size))))
        biases = get_scope_variable(
            name_scope=name_scope, var='biases', shape=[col_size],
            initializer=tf.constant_initializer(0.0))

        self.weights_all.append(weights)
        self.biases_all.append(biases)
        variable_summaries(weights)
        variable_summaries(biases)

        with tf.variable_scope(name_scope) as scope:
            input = tf.matmul(last_hidden, weights) + biases
            if act == None:
                output = input
            else:
                output = act(input, name='output')

        self.inputs.append(input)
        self.hidden.append(output)
        return output

    @property
    def num_of_layers(self):
        return self._num_of_layers

    @property
    def hidden_layers(self):
        if self.hidden is None:
            self.hidden, self.inputs, = [], []
            self.weights_all, self.biases_all = [], []

            last_hidden = self.x
            if self.cov == 1:
                y_conv, self._drouput, self.hidden, self.inputs = deepnn(self.x)
            elif self.cov == 2:
                y_c, self.hidden, self.inputs = multi_layer_perceptron(
                    self.x, self.input_size, self.num_of_classes,
                    self.layers[0], self.layers[1])
            else:
                self._drouput = 'dr'
                for i in range(1, len(self.all_layers)):
                    name_scope = 'hidden' + str(i - 1)
                    row_size = self.all_layers[i - 1]
                    col_size = self.all_layers[i]
                    act = self.act
                    last_hidden = self.initilizae_layer(name_scope, row_size,
                                                        col_size, act,
                                                        last_hidden)
                name_scope = 'final_layer'
                row_size, col_size = self.layers[-1], self.num_of_classes
                act = tf.nn.softmax
                last_hidden = self.initilizae_layer(name_scope, row_size,
                                                    col_size, act, last_hidden)
        return self.hidden

    @lazy_property
    def prediction(self):
        logits = self.hidden_layers[-1]
        return logits

    @lazy_property
    def drouput(self):
        return self._drouput

    @property
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate=self.lr_local).minimize(
                            self.cross_entropy)
        return optimizer

    @lazy_property
    def x(self):
        return tf.placeholder(tf.float32, shape=[None, self.input_size], name='x')

    @lazy_property
    def labels(self):
        return tf.placeholder(tf.float32, shape=[None, self.num_of_classes], name='y_true')

    @lazy_property
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @lazy_property
    def cross_entropy(self):
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.labels * tf.log(tf.clip_by_value(self.prediction, 1e-50, 1.0)), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy

    @property
    def save_file(self):
        return self._save_file

    def inference(self, data):
        """Return the predication of the network with the given data"""
        with tf.Session() as sess:
            self.saver.restore(sess, './' + self.save_file)
            feed_dict = { self.x: data }
            pred = sess.run(self.prediction, feed_dict=feed_dict)
        return pred

    def inference_default(self, data):
        session = tf.get_default_session()
        feed_dict = {self.x: data}
        pred = session.run(self.prediction, feed_dict=feed_dict)
        return pred

    def get_layer_with_inference(self, data, layer_index, epoch_index):
        """Return the layer act's values with the results of the network"""
        with tf.Session() as sess:
            self.savers[epoch_index].restore(sess, './' + self.save_file + str(epoch_index))
            feed_dict = {self.hidden_layers[layer_index]: data[:, 0:self.hidden_layers[layer_index]._shape[1]]}
            pred, layer_values = sess.run([self.prediction, self.hidden_layers[layer_index]], feed_dict=feed_dict)
        return pred, layer_values

    def calc_layer_values(self, X, layer_index):
        """Return the layer's values"""
        with tf.Session() as sess:
            self.savers[-1].restore(sess, './' + self.save_file)
            feed_dict = {self.x: X}
            layer_values = sess.run(self.hidden_layers[layer_index], feed_dict=feed_dict)
        return layer_values

    def update_weights_and_calc_values_temp(self, d_w_i_j, layer_to_perturbe, i, j, X):
        """Update the weights of the given layer cacl the output and return it to the original values"""
        if layer_to_perturbe + 1 >= len(self.hidden_layers):
            scope_name = 'softmax_linear'
        else:
            scope_name = "hidden" + str(layer_to_perturbe)
        weights = get_scope_variable(scope_name, "weights", shape=None, initializer=None)
        session = tf.get_default_session()
        weights_values = weights.eval(session=session)
        weights_values_pert = weights_values
        weights_values_pert[i, j] += d_w_i_j
        set_value(weights, weights_values_pert)
        feed_dict = {self.x: X}
        layer_values = session.run(self.hidden_layers[layer_to_perturbe], feed_dict=feed_dict)
        set_value(weights, weights_values)
        return layer_values

    def update_weights(self, d_w0, layer_to_perturbe):
        """Update the weights' values of the given layer"""
        weights = get_scope_variable("hidden" + str(layer_to_perturbe), "weights", shape=None, initializer=None)
        session = tf.get_default_session()
        weights_values = weights.eval(session=session)
        set_value(weights, weights_values + d_w0)

    def get_wights_size(self, layer_to_perturbe):
        """Return the size of the given layer"""
        weights = get_scope_variable("hidden" + str(layer_to_perturbe), "weights", shape=None, initializer=None)
        return weights._initial_value.shape[1].value, weights._initial_value.shape[0].value

    def get_layer_input(self, layer_to_perturbe, X):
        """Return the input of the given layer for the given data"""
        session = tf.get_default_session()
        inputs = self.inputs[layer_to_perturbe]
        feed_dict = {self.x: X}
        layer_values = session.run(inputs, feed_dict=feed_dict)
        return layer_values
