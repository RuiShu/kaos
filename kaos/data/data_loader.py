import numpy as np
import pickle as pkl
import os, urllib, gzip
import scipy.io

class StandardDataLoader(object):
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def get_training_batch(self, batchsize=None):
        self.batchsize = self.batchsize if batchsize is None else batchsize
        idx = np.random.choice(len(self.x_train), self.batchsize, replace=False)
        return [self.x_train[idx], self.y_train[idx]]

    def get_validation_batch(self, batchsize=None):
        self.batchsize = self.batchsize if batchsize is None else batchsize
        idx = np.random.choice(len(self.x_valid), self.batchsize, replace=False)
        return [self.x_valid[idx], self.y_valid[idx]]

    def get_test_batch(self, batchsize=None):
        self.batchsize = self.batchsize if batchsize is None else batchsize
        idx = np.random.choice(len(self.x_test), self.batchsize, replace=False)
        return [self.x_test[idx], self.y_test[idx]]

    def get_training_set(self):
        return [self.x_train, self.y_train]

    def get_validation_set(self):
        return [self.x_valid, self.y_valid]

    def get_test_set(self):
        return [self.x_test, self.y_test]

    def _preprocess(self):
        # Some type of preprocessing?
        raise NotImplementedError

class MnistLoader(StandardDataLoader):
    def __init__(self, batchsize=100):
        super(MnistLoader, self).__init__(batchsize)
        self._load_mnist()
        self.ssl_compatible = False

    @staticmethod
    def _download_mnist():
        folder = os.path.join('data', 'mnist_real')
        data_loc = os.path.join(folder, 'mnist.pkl.gz')
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(data_loc):
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print "Downloading data from:", url
            urllib.urlretrieve(url, data_loc)
        return data_loc

    def _load_mnist(self):
        f = gzip.open(self._download_mnist(), 'rb')
        train, valid, test = pkl.load(f)
        f.close()
        self.x_train, self.y_train = train[0], train[1]
        self.x_valid, self.y_valid = valid[0], valid[1]
        self.x_test, self.y_test = test[0], test[1]

    def _binarize(self, data, seed=None):
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)
        v = np.random.uniform(0, 1, data.shape)
        data = (data > v).astype(np.float32)
        if seed is not None:
            np.random.set_state(state)
        return data

    def binarize(self, seed=42):
        state = np.random.get_state()
        np.random.seed(seed)
        v = np.random.uniform(0, 1, self.x_train.shape)
        self.x_train = (self.x_train > v).astype(np.float32)
        v = np.random.uniform(0, 1, self.x_valid.shape)
        self.x_valid = (self.x_valid > v).astype(np.float32)
        v = np.random.uniform(0, 1, self.x_test.shape)
        self.x_test = (self.x_test > v).astype(np.float32)
        np.random.set_state(state)
        return self

    def convert_to_ssl(self, n_labeled_samples, seed):
        assert self.ssl_compatible, "Compatability flag is off"
        state = np.random.get_state()
        np.random.seed(seed)
        x, y = self.balanced_sampler(self.x_train, self.y_train, n_labeled_samples)
        self.x_label = x
        self.y_label = y
        np.random.set_state(state)
        return self

    @staticmethod
    def balanced_sampler(x, y, n):
        assert (n % 10 == 0), "sample size must be divisible by 10"
        x_sample = np.empty((0, 784))
        y_sample = np.empty((0, 10))
        eye = np.eye(10)
        for i in xrange(10):
            idx = y == i
            xi, yi = x[idx], y[idx]
            shuffle = np.random.permutation(len(xi))
            xi, yi = xi[shuffle], eye[yi[shuffle]]
            x_sample = np.vstack((x_sample, xi[:n//10]))
            y_sample = np.vstack((y_sample, yi[:n//10]))
        return x_sample, y_sample

    def get_labeled_training_set(self):
        return self.x_label, self.y_label

    def get_labeled_training_batch(self, batchsize=None):
        self.batchsize = self.batchsize if batchsize is None else batchsize
        idx = np.random.choice(len(self.x_label), self.batchsize, replace=False)
        return [self.x_label[idx], self.y_label[idx]]
