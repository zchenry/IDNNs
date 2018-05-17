import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import os
import sys
import tensorflow as tf

def load_data(dataset):
    print('Loading Data...')
    if dataset.split('/')[-1] == 'MNIST':
        pass
        #data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0]) + "/data/MNIST_data/", one_hot=True)
        #data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis=0)
        #data_sets.labels = np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis=0)
    else:
        d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), dataset + '.mat'))
        xs = d['F']
        ys = d['y'].T
        data_sets = np.append(xs, ys, axis=1)
    return data_sets

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)
