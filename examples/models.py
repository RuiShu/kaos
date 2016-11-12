from kaos.bayes import BayesNet
from kaos.distributions import log_bernoulli, log_normal, kl_normal
from kaos.utils import rename
from keras.layers import Input, merge
from keras import backend as K
import numpy as np

def concat(*args):
    return merge(args, mode='concat')

class Base(BayesNet):
    def _define_log_importance_likelihood(self, x, x_param, loss):
        log_imp_likelihood = -loss(x, x_param)
        self._log_importance_likelihood = K.function([x, K.learning_phase()], log_imp_likelihood)

    def log_importance_likelihood(self, data_input, data_output, n_samples):
        assert self._compute_log_likelihood, ("Must set to True "
                                              "during compilation")
        x = data_input
        ln_imp = np.empty((len(x), 0))

        for i in xrange(n_samples):
            sample = self._log_importance_likelihood([x, 0]).reshape(-1,1)
            ln_imp = np.hstack((ln_imp, sample))

        return ln_imp

class VAE(Base):
    def _define_io_loss(self):
        x = Input(shape=self.shape['x'])
        q_net, p_net = self.q_net, self.p_net
        q, p, s = {}, {}, {}
        # inference
        q['z'], s['z'] = q_net['z'](x, sample=True)
        # generation
        p['z'] = p_net['z']()
        p['x'] = p_net['x'](s['z'])

        @rename('loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        if self._compute_log_likelihood:
            self._define_log_importance_likelihood(x, p['x'], loss)

        return self._standardize_io_loss(x, p['x'], loss)

class LadderVAE(Base):
    def _define_io_loss(self):
        x = Input(shape=self.shape['x'])
        u_net, l_net = self.u_net, self.l_net
        q_net, p_net = self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['z1'] = u_net['z1'](x)
        u['z2'] = u_net['z2'](u['z1'])
        # likelihood
        l['z1'] = l_net['z1'](u['z1'])
        # generation and inference
        p['z2'] = p_net['z2']()
        q['z2'], s['z2'] = q_net['z2'](u['z2'], sample=True)
        p['z1'] = p_net['z1'](s['z2'])
        q['z1'], s['z1'] = q_net['z1'](l['z1'], p['z1'], sample=True)
        p['x'] = p_net['x'](s['z1'])

        @rename('loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss

        if self._compute_log_likelihood:
            self._define_log_importance_likelihood(x, p['x'], loss)

        return self._standardize_io_loss(x, p['x'], loss)

class AuxiliaryVAE(Base):
    def _define_io_loss(self):
        x = Input(shape=self.shape['x'])
        q_net, p_net = self.q_net, self.p_net
        q, p, s = {}, {}, {}

        # inference
        q['a'], s['a'] = q_net['a'](x, sample=True)
        q['z'], s['z'] = q_net['z'](concat(x, s['a']), sample=True)
        # generation
        p['z'] = p_net['z']()
        p['x'] = p_net['x'](s['z'])
        p['a'] = p_net['a'](concat(x, s['z']))

        @rename('loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            loss += log_normal(s['a'], q['a']) - log_normal(s['a'], p['a'])
            return loss

        if self._compute_log_likelihood:
            self._define_log_importance_likelihood(x, p['x'], loss)

        return self._standardize_io_loss(x, p['x'], loss)
