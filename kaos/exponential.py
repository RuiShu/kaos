"""Deprecated. Alternative to softplus activation for computing variance.
"""
from keras.layers import merge
from keras import backend as K
import numpy as np


def sampling(z_par):
    batch_size = z_par.shape[0]
    dim = z_par.shape[1]//2
    z_mu, z_lv = z_par[:, :dim], z_par[:, dim:]
    epsilon = K.random_normal(shape=(batch_size, dim))
    return z_mu + K.exp(z_lv/2) * epsilon

def sampling_shape(input_shape):
    assert len(input_shape) == 2  # only valid for 2D tensors
    assert input_shape[1] % 2 == 0 # only valid if last dim is even
    return (input_shape[0], input_shape[1]//2)

gaussian_sampler = Lambda(sampling, sampling_shape)

def log_normal(x, p=0):
    if p is 0:
        mu, lv = 0, 0
    else:
        dim = p.shape[1]//2
        mu, lv = p[:, :dim], p[:, dim:]
    return - 0.5 * K.sum(K.log(2*np.pi) + lv + K.square(x - mu) * K.exp(-lv), axis=-1)

def kl_normal(q, p=0):
    dim = q.shape[1]//2
    qmu, qlv = q[:, :dim], q[:, dim:]
    if p is 0:
        pmu, plv = 0, 0
    else:
        pmu, plv = p[:, :dim], p[:, dim:]
    return 0.5 * K.sum(plv - qlv + K.exp(qlv - plv) + K.square(qmu - pmu) * K.exp(-plv) - 1, axis=-1)

