from kaos.data import MnistLoader
from kaos.data import StandardDataLoader
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

class MnistVAE(MnistLoader):
    def __init__(self, batchsize):
        super(MnistVAE, self).__init__(batchsize)
        self.binarize()

    def get_training_batch(self):
        x, y = super(MnistVAE, self).get_training_batch()
        return x, x

    def get_training_set(self):
        x, y = super(MnistVAE, self).get_training_set()
        return x, x

    def get_validation_set(self):
        x, y = super(MnistVAE, self).get_validation_set()
        return x, x

    def get_test_set(self):
        x, y = super(MnistVAE, self).get_test_set()
        return x, x

class MnistSemiSupervised(StandardDataLoader):
    def __init__(self, nb_data, batchsize):
        super(MnistSemiSupervised, self).__init__(batchsize)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # subsample
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        x_u, y_u = x_train, np.zeros((len(x_train), 10))
        x_l, y_l = self.balanced_sampler(x_train, y_train, nb_data)
        # Convert class vectors to binary class matrices.
        y_l = np_utils.to_categorical(y_l, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        self.x_train, self.y_train = x_u, y_u
        self.x_label, self.y_label = x_l, y_l
        self.x_valid, self.y_valid = x_test, y_test

    @staticmethod
    def balanced_sampler(x, y, n):
        assert (n % 10 == 0), "sample size must be divisible by 10"
        y = y.reshape(-1)
        x_sample = np.empty((0,) + x.shape[1:])
        y_sample = np.empty((0,))

        for i in xrange(10):
            idx = y == i
            xi, yi = x[idx], y[idx]
            shuffle = np.random.permutation(len(xi))
            xi, yi = xi[shuffle], yi[shuffle]
            x_sample = np.vstack((x_sample, xi[:n//10]))
            y_sample = np.hstack((y_sample, yi[:n//10]))
        return x_sample, y_sample.astype(int)

    def get_labeled_training_batch(self, batchsize=None):
        self.batchsize = self.batchsize if batchsize is None else batchsize
        idx = np.random.choice(len(self.x_label), self.batchsize, replace=False)
        return [self.x_label[idx], self.y_label[idx]]

    def get_training_batch(self):
        xl, yl = self.get_labeled_training_batch()
        xu, yu = super(MnistSemiSupervised, self).get_training_batch()
        return [xl, yl, xu, yu], [yl, xl, xu]

    def get_training_set(self):
        xl, yl = self.x_label, self.y_label
        xu, yu = self.x_train, self.y_train
        return [xl, yl, xu, yu], [yl, xl, xu]

    def get_validation_set(self):
        x, y = super(MnistSemiSupervised, self).get_validation_set()
        return [x, y, x, y], [y, x, x]
