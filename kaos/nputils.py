import numpy as np

def log_sum_exp(x, axis=-1):
    a = x.max(axis=axis, keepdims=True)
    out = a + np.log(np.sum(np.exp(x - a), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def kl_normal(q, p=0):
    dim = q.shape[1]//2
    qmu, qlv = q[:, :dim], q[:, dim:]
    if p is 0:
        pmu, plv = 0, 0
    else:
        pmu, plv = p[:, :dim], p[:, dim:]
    return 0.5 * np.sum(plv - qlv + np.exp(qlv - plv) + np.square(qmu - pmu) * np.exp(-plv) - 1, axis=-1)

def gaussian_split(par):
    assert len(par.shape) == 2
    dim = par.shape[1]/2
    return par[:, :dim], par[:, dim:]

def gaussian_sample(par):
    assert len(par.shape) == 2
    batch_size, dim = par.shape[0], par.shape[1]/2
    mu, lv = par[:, :dim], par[:, dim:]
    eps = np.random.randn(batch_size, dim)
    return mu + np.exp(lv/2) * eps
