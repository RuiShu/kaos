from keras.layers import merge
from keras import backend as K
import numpy as np

"""Sampling and probability density functions based on softplus"""
def sampling(z_par):
    z_mu, z_var = z_par
    epsilon = K.random_normal(shape=z_mu.shape)
    return z_mu + K.sqrt(z_var) * epsilon

def sampling_shape(input_shape):
    input_shape = input_shape[0]
    assert len(input_shape) == 2  # only valid for 2D tensors
    return input_shape

def gaussian_sampler(z_par):
    return merge(z_par, mode=sampling, output_shape=sampling_shape)

def log_normal(x, p=None, eps=0.0):
    if p is None:
        mu, var = 0, 1
    else:
        mu, var = p
	var += eps
    return - 0.5 * K.sum(K.log(2*np.pi) + K.log(var) + K.square(x - mu) / var, axis=-1)

def kl_normal(q, p=None):
    qmu, qvar = q
    if p is None:
        pmu, pvar = 0, 1
    else:
        pmu, pvar = p
    return 0.5 * K.sum(K.log(pvar) - K.log(qvar) + qvar/pvar + K.square(qmu - pmu) / pvar - 1, axis=-1)
