import _pickle as cPickle
import multiprocessing
import os
import sys
import numpy as np
from joblib import Parallel, delayed
import idnns.networks.network as nn
from idnns.information import information_process  as inn
from idnns.plots import plot_figures as plt_fig
from idnns.networks.utils import load_data
import argparse
NUM_CORES = multiprocessing.cpu_count()

def get_default_parser(snaps=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--layers', nargs='+', type=int, default=[[5, 3]])
    parser.add_argument('--name', default='net', type=str)
    parser.add_argument('--dataset', default='var_u', type=str)
    parser.add_argument('--snaps', default=2, type=int)
    parser.add_argument('--nbins', default=30, type=int)
    parser.add_argument('--act', default=0, type=int) # 0: tanh, 1: relu
    parser.add_argument('--cov', default=0, type=int)
    parser.add_argument('--percents', nargs='+', type=int, default=[80])
    args = parser.parse_args()
    return args

class informationNetwork():
    def __init__(self, snaps=None, args=None):
        if args == None:
            args = get_default_parser(snaps)
        self.cov = args.cov
        self.epochs = args.epochs
        self.lr = args.lr
        self.batchsize = args.batchsize
        self.act = args.act
        self.repeats = args.repeats
        self.nbins = args.nbins
        self.datafile = 'data/{}'.format(args.dataset)
        self.layers = args.layers
        self.percents = args.percents
        self.snap_epochs = np.unique(
            np.logspace(np.log2(1), np.log2(args.epochs), args.snaps,
                        dtype=int, base=2)) - 1

        self.data_sets = load_data(self.datafile)

        # create arrays for saving the data
        self.ws, self.grads, self.information, \
            self.models, self.names, self.networks, self.weights = [
                [[[[None] for k in range(len(self.percents))]
                  for j in range(len(self.layers))]
                 for i in range(self.repeats)] for _ in range(7)]

        self.loss_train, self.loss_test, self.test_error, \
                self.train_error, self.l1_norms, self.l2_norms = \
            [np.zeros((self.repeats, len(self.layers),
                       len(self.percents), len(self.snap_epochs)))
             for _ in range(6)]

        params = {
            'data': args.dataset,
            'layers': args.layers,
            'epochs': args.epochs,
            'batch': args.batchsize,
            'lr': args.lr,
            'repeats': args.repeats,
            'percents': self.percents }

        name_to_save = '|'.join([str(i) + '=' + str(params[i]) for i in params])
        self.name_to_save = name_to_save
        params['directory'] = self.name_to_save

        params['snapepochs'] = len(self.snap_epochs)
        params['CPUs'] = NUM_CORES
        params['snapepochs'] = self.snap_epochs

        self.params = params
        self.is_trained = False

    def save_data(self, parent_dir='jobs/', file_to_save='data.pickle'):
        directory = '{0}/{1}{2}/'.format(
                os.getcwd(), parent_dir, self.params['directory'])

        data = { 'information': self.information,
                 'test_error': self.test_error,
                 'train_error': self.train_error,
                 'var_grad_val': self.grads,
                 'loss_test': self.loss_test,
                 'loss_train': self.loss_train,
                 'params': self.params,
                 'l1_norms': self.l1_norms,
                 'weights': self.weights,
                 'ws': self.ws }

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.dir_saved = directory
        with open(self.dir_saved + file_to_save, 'wb') as f:
            cPickle.dump(data, f, protocol=2)

    def run_network(self):
        '''
        results = Parallel(n_jobs=NUM_CORES, verbose=10000)( [
            delayed(nn.train_and_calc_inf_network)
            (i, j, k, self.layers[j], self.epochs, self.lr,
             self.batchsize, self.snap_epochs, self.data_sets,
             self.act, self.percents, self.nbins, self.cov)
            for i in range(len(self.percents))
            for j in range(len(self.layers))
            for k in range(self.repeats) ] )

        '''
        results = [nn.train_and_calc_inf_network(
            i, j, k, self.layers[j], self.epochs, self.lr, self.batchsize,
            self.snap_epochs, self.data_sets, self.act, self.percents,
            self.nbins, self.cov)
                   for i in range(len(self.percents))
                   for j in range(len(self.layers))
                   for k in range(self.repeats)]

        # Extract all the measures
        for i in range(len(self.percents)):
            for j in range(len(self.layers)):
                for k in range(self.repeats):
                    index = i * len(self.layers) * self.repeats + \
                            j * self.repeats + k
                    current_network = results[index]
                    self.networks[k][j][i] = current_network
                    self.ws[k][j][i] = current_network['ws']
                    self.weights[k][j][i] = current_network['weights']
                    self.information[k][j][i] = current_network['information']
                    self.grads[k][i][i] = current_network['gradients']
                    self.test_error[k, j, i, :] = current_network['test_prediction']
                    self.train_error[k, j, i, :] = current_network['train_prediction']
                    self.loss_test[k, j, i, :] = current_network['loss_test']
                    self.loss_train[k, j, i, :] = current_network['loss_train']
        self.is_trained = True

    def print_information(self):
        for val in self.params: print('{:>10s}: {}'.format(val, self.params[val]))

    def plot_network(self):
        str_names = [[self.dir_saved]]
        save_name = 'jobs/run'
        plt_fig.plot_figures(str_names, 0, save_name)
        plt_fig.plot_figures(str_names, 2, save_name)
        plt_fig.plot_figures(str_names, 4, save_name)
