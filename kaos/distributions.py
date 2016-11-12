from keras.layers import Lambda, merge
from keras import backend as K
from ladder import *
from softplus import *

import numpy as np
import pickle as pkl
import os, urllib, gzip

class Distribution(object):
    """Distribution class for computing distribution parameters.

    This is where the variational framework meets neural network function
    approximators. The power of VAEs stems from the use of NNs to determine the
    parameters of conditional distributions.

    Example:
    A conditional Gaussian

    """
    def __init__(self, callback=None, bind=False, sampler=None):
        self.callback = None
        self.sampler = sampler
        if callback is not None:
            self.set_callback(callback, bind)

    def set_callback(self, callback, bind=False):
        if bind:
            self.callback = callback.__get__(self, Distribution)
        else:
            self.callback = callback

    def __call__(self, *given, **kwargs):
        param = self.callback(*given)
        if kwargs.get('sample', False):
            if self.sampler is None:
                raise Exception('No sampler provided. ' +
                                'Cannot sample distribution')
            return param, self.sampler(param)
        else:
            return param

# """Probability density functions"""
def log_bernoulli(x, p):
    return -K.sum(K.binary_crossentropy(p, x), axis=-1)
