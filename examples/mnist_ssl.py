from data import MnistSemiSupervised as Mnist
from keras.layers import Dense, Activation, Input, Lambda, merge
from keras.models import Sequential
from keras import backend as K
from kaos.distributions import Distribution as Dist
from kaos.distributions import gaussian_sampler
from kaos.distributions import log_bernoulli, log_normal
from kaos.bayes import BayesNet
from kaos.callbacks import LossLog, NegativeLogLikelihood
import numpy as np

class VAE(BayesNet):
    def xy_graph(self, x, y):
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        uz = u_net['z'](concat(x, y))
        qz, sz = q_net['z'](uz, sample=True)
        px = p_net['x'](concat(sz, y))
        return qz, sz, px

    @staticmethod
    def labeled_loss(x, qz, sz, px):
        loss = -log_bernoulli(x, px)
        loss += log_normal(sz, qz) - log_normal(sz, (0, 1))
        return loss

    def _define_io_loss_x(self):
        u, p, q, s = {}, {}, {}, {}
        x, y = Input(shape=(784,)), Input(shape=(10,))
        u['x'] = self.u_net['x'](x)
        q['y'] = self.q_net['y'](u['x'])

        # create one-hots
        s['y'] = []
        for i in xrange(10):
            l = Lambda(lambda x: x + K.variable(np.eye(10)[i]), (10,))
            s['y'] += [l(y)]

        q['z'], s['z'], p['x'] = [], [], []
        for i in xrange(10):
            a, b, c = self.xy_graph(x, s['y'][i])
            q['z'] += [a]
            s['z'] += [b]
            p['x'] += [c]

        def x_loss(x, x_param):
            loss = K.categorical_crossentropy(q['y'], q['y'])
            for i in xrange(10):
                loss += q['y'][:, i] * self.labeled_loss(x, q['z'][i], s['z'][i], p['x'][i])
            return loss

        return self._standardize_io_loss([x, y], p['x'][0], x_loss)

    def _define_io_loss_xy(self):
        u, p, q, s = {}, {}, {}, {}
        x, y = Input(shape=(784,)), Input(shape=(10,))
        q['z'], s['z'], p['x'] = self.xy_graph(x, y)
        u['x'] = self.u_net['x'](x)
        q['y'] = self.q_net['y'](u['x'])

        def alpha_loss(y, y_param):
            return K.categorical_crossentropy(q['y'], y)

        def xy_loss(x, x_param):
            return self.labeled_loss(x, q['z'], s['z'], p['x'])

        self._predict = K.function([x, K.learning_phase()], q['y'])
        return self._standardize_io_loss([x, y],
                                         [q['y'], p['x']],
                                         [alpha_loss, xy_loss])

    def _define_io_loss(self):
        x = self._define_io_loss_x()
        xy = self._define_io_loss_xy()
        return self._zip_io_losses(xy, x)

    def log_likelihood(self, data, n_samples):
        """Not actually log likelihood. Secretly computing accuracy as NLL proxy.
        Basically retrofitting the NegativeLogLikelihood callback.

        TODO: Implement generic metric callback
        """
        inputs, outputs = data
        x, y = inputs[:2]
        y_pred = self._predict([x, 0])
        ans = np.mean(y_pred.argmax(axis=-1) == y.argmax(axis=-1))
        return -ans, -ans

def concat(*args):
    return merge(args, mode='concat')

def muv_call(self, x):
    mu = self.mu(x)
    var = self.var(x)
    return (mu, var)

def net_call(self, x):
    return self.net(x)

u_net, l_net, p_net, q_net = {}, {}, {}, {}
q_net['z'] = Dist(callback=muv_call, bind=True, sampler=gaussian_sampler)
p_net['z'] = Dist(callback=lambda: (0, 1), bind=False, sampler=gaussian_sampler)
p_net['x'] = Dist(callback=net_call, bind=True, sampler=gaussian_sampler)

u_net['x'] = Sequential([Dense(256, input_dim=784),
                         Activation('relu'),
                         Dense(256),
                         Activation('relu')])

q_net['y'] = Sequential([Dense(10, input_dim=256),
                         Activation('softmax')])

u_net['z'] = Sequential([Dense(256, input_dim=794),
                         Activation('relu'),
                         Dense(256),
                         Activation('relu')])

q_net['z'].mu = Sequential([Dense(50, input_dim=256)])

q_net['z'].var = Sequential([Dense(50, input_dim=256),
                             Activation('softplus')])

p_net['x'].net = Sequential([Dense(256, input_dim=60),
                             Activation('relu'),
                             Dense(256),
                             Activation('relu'),
                             Dense(784),
                             Activation('sigmoid')])

mnist = Mnist(100, 100)
vae = VAE(u_net=u_net, q_net=q_net, p_net=p_net)
vae.compile('adam', loss_weights=[1.0, 100./50000., 1.0])

dataloader = Mnist(100, 100)
losslog = LossLog()
nll = NegativeLogLikelihood(dataloader,
                            n_samples=1,
                            run_every=1,
                            run_training=True,
                            run_validation=True,
                            display_epoch=True,
                            end_line=True)
vae.fit(dataloader,
        nb_epoch=1000,
        iter_per_epoch=600,
        callbacks=[losslog, nll],
        verbose=1)
