import multiprocessing
import os
import sys
import warnings
import numpy as np
import tensorflow as tf
from idnns.information import information_process  as inn
from idnns.networks import model as mo
warnings.filterwarnings('ignore')

summaries_dir = 'summaries'
NUM_CORES = multiprocessing.cpu_count()

def build_model(*args):
    return mo.Model(*args)

def train_and_calc_inf_network(i, j, k, layers, num_of_ephocs, lr_local,
                               batchsize, indexes, data_sets,
                               model_type, percent_of_train,
                               nbins, cov):
    """Train the network and calculate its information"""
    network_name = '{0}_{1}_{2}'.format(i, j, k)
    print ('Training network  - {0}'.format(network_name))
    network = train_network(layers, num_of_ephocs, lr_local, batchsize,
                            indexes, data_sets, model_type,
                            percent_of_train,
                            network_name, cov)
    network['information'] = []

    print('Calculating the infomration')
    infomration = np.array(
        [inn.get_information(network['ws'], data_sets[:, :-1],
                             data_sets[:, -1:], nbins,
                             network['model'], layers)])
    network['information'] = infomration
    print('Successfully calculated the infomration !!!')
    return network

def exctract_activity(sess, batch_points_all, model, data_sets):
    """Get the act values of the layers for the input"""
    w_temp = []
    for i in range(0, len(batch_points_all) - 1):
        batch = data_sets[int(batch_points_all[i]):int(batch_points_all[i + 1])]
        batch_xs = batch[:, :-1]
        batch_ys = batch[:, -1:]
        feed_dict_temp = { model.x: batch_xs, model.labels: batch_ys }
        w_temp_local = sess.run([ model.hidden_layers ],
                                feed_dict=feed_dict_temp)
        for s in range(len(w_temp_local[0])):
            if i == 0:
                w_temp.append(w_temp_local[0][s])
            else:
                w_temp[s] = np.concatenate((w_temp[s], w_temp_local[0][s]),
                                           axis=0)
    return w_temp

def print_accuracy(batch_points_test, data_sets,
                   model, sess, j, acc_train_array):
    """Calc the test acc and print the train and test accuracy"""
    acc_array = []
    N = int(data_sets.shape[0] * 0.8)
    data = data_sets[N:]
    for i in range(0, len(batch_points_test) - 1):
        batch = data_sets[int(batch_points_test[i]):int(batch_points_test[i + 1])]
        batch_xs = batch[:, :-1]
        batch_ys = batch[:, -1:]
        feed_dict_temp = { model.x: batch_xs, model.labels: batch_ys }
        acc = sess.run([model.accuracy], feed_dict=feed_dict_temp)
        acc_array.append(acc)
    print ('Epoch {0} - Test Accuracy: {1:.3f} Train Accuracy: {2:.3f}'.format(
            j, np.mean(np.array(acc_array)), np.mean(np.array(acc_train_array))))

def train_network(layers, num_of_ephocs, lr_local, batchsize, indexes,
                  data_sets, model_type, percent_of_train, name, cov):
    tf.reset_default_graph()
    ws, estimted_label, gradients, infomration, models, weights = [
            [None] * len(indexes) for _ in range(6)]
    loss_func_test, loss_func_train, test_prediction, train_prediction = [
            np.zeros((len(indexes))) for _ in range(4)]

    input_size = data_sets.shape[1] - 1
    batchsize = np.min([batchsize, int(data_sets.shape[0] * 0.8)])

    N = data_sets.shape[0]
    batch_points = np.rint(np.arange(0, int(N * 0.8) + 1, batchsize)).astype(dtype=np.int32)
    batch_points_test = np.rint(np.arange(0, int(N * 0.2) + 1, batchsize)).astype(dtype=np.int32)
    batch_points_all = np.rint(np.arange(0, N + 1, batchsize)).astype(dtype=np.int32)

    if N not in batch_points_all:
        batch_points_all = np.append(batch_points_all, [N])
    if int(N * 0.8) not in batch_points:
        batch_points = np.append(batch_points, [N * 0.8])
    if int(N * 0.2) not in batch_points_test:
        batch_points_test = np.append(batch_points_test, [N * 0.2])

    model = build_model(input_size, layers, 1,
                        lr_local, name, model_type, cov)
    optimizer = model.optimize
    saver = tf.train.Saver(max_to_keep=0)
    init = tf.global_variables_initializer()
    grads = tf.gradients(model.cross_entropy, tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(init)
        # Go over the epochs
        k = 0
        acc_train_array = []
        for j in range(0, num_of_ephocs):
            epochs_grads = []
            if j in indexes:
                ws[k] = exctract_activity(sess, batch_points_all,
                                          model, data_sets)

            if np.mod(j, 100) == 1:
                print_accuracy(batch_points_test, data_sets, model,
                               sess, j, acc_train_array)

            # Go over the batch_points
            acc_train_array = []
            current_weights = [[] for _ in range(len(model.weights_all))]
            for i in range(0, len(batch_points) - 1):
                batch = data_sets[int(batch_points[i]):int(batch_points[i + 1])]
                batch_xs = batch[:, :-1]
                batch_ys = batch[:, -1:]
                feed_dict = { model.x: batch_xs, model.labels: batch_ys }
                _, tr_err = sess.run( [ optimizer, model.accuracy ],
                                      feed_dict=feed_dict )
                acc_train_array.append(tr_err)

                if j in indexes:
                    epochs_grads_temp, loss_tr, weights_local = sess.run(
                        [grads, model.cross_entropy, model.weights_all],
                        feed_dict=feed_dict)
                    epochs_grads.append(epochs_grads_temp)
                    for ii in range(len(current_weights)):
                        current_weights[ii].append(weights_local[ii])

            if j in indexes:
                gradients[k] = epochs_grads
                current_weights_mean = []
                for ii in range(len(current_weights)):
                    current_weights_mean.append(
                        np.mean(np.array(current_weights[ii]), axis=0))
                weights[k] = current_weights_mean

                # Save the model
                write_meta = True if k == 0 else False
                k += 1

    network = {}
    network['ws'] = ws
    network['weights'] = weights
    network['test_prediction'] = test_prediction
    network['train_prediction'] = train_prediction
    network['loss_test'] = loss_func_test
    network['loss_train'] = loss_func_train
    network['gradients'] = gradients
    network['model'] = model
    return network
