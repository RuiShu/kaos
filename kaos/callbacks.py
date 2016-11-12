from keras.callbacks import Callback
from keras import backend as K
from nputils import log_sum_exp
from utils import tuplify, listify
import numpy as np
import sys
import os

def _print(stream, string):
    print >>stream, string,
    if stream is not sys.stdout:
        print string,

def _pretty_print(stream, name, val):
    _print(stream, '{0:s}: {1:9.3e} ;'.format(name, val))

class NegativeLogLikelihood(Callback):
    """Computes the negative log likelihood.

    This class the currently bloated and handles many responsibilities,
    including learning rate step decay, early-stopping, and model-saving.
    TODO: Split class into smaller classes
    """
    def __init__(self, dataloader, n_samples=100, run_every=1,
                 run_training=False, run_validation=False, run_test=False,
                 display_nelbo=False, display_wait=False, display_epoch=False,
                 display_best_val=False,
                 step_decay=1.0, patience=np.inf,
                 save_model_path=None,
                 fh=sys.stdout, end_line=False):
        self.dataloader = dataloader
        self.run_every = run_every
        self.step_decay = step_decay
        self.patience = patience
        self.run_training = run_training
        self.run_validation = run_validation
        self.run_test = run_test
        self.display_nelbo = display_nelbo
        self.display_wait = display_wait
        self.display_epoch = display_epoch
        self.display_best_val = display_best_val or run_test
        self.n_samples = n_samples
        self.fh = fh
        self.end_line = end_line
        self.save_model_path = save_model_path
        if save_model_path is not None:
            self.model_saver = ModelSaver(filepath=save_model_path, fh=fh)

    def _set_model(self, model):
        self.model = model
        if self.save_model_path is not None:
            self.model_saver.model = model

    def on_train_begin(self, logs={}):
        if self.run_validation:
            self.best_val_nll = np.inf
        if self.run_test:
            self.test_nll = np.inf
            self.test_nelbo = np.inf
        self.wait = 0
        self.lr_patience = self.patience/2

    def _compute_negative_log_likelihood(self, data):
        ll, elbo = self.model.log_likelihood(data, n_samples=self.n_samples)
        nll = -np.mean(ll)
        nelbo = -np.mean(elbo)
        return nll, nelbo

    def _log(self, name, nll, nelbo):
        _pretty_print(self.fh, name+'NLL', nll)
        if self.display_nelbo:
            _pretty_print(self.fh, name+'NELBO', nelbo)

    def _check_val_nll(self, val_nll):
        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
            self.wait = 0
            self.lr_patience = self.patience/2
            return self.best_val_nll, True
        else:
            self.wait += 1
            return self.best_val_nll, False

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.run_every != 0:
            if self.display_epoch:
                _pretty_print(self.fh, 'epoch', epoch)

            if self.end_line:
                _print(self.fh, '\n')
            return

        if self.run_training:
            tr_nll, tr_nelbo = self._compute_negative_log_likelihood(
                self.dataloader.get_training_set())
            self._log('tr', tr_nll, tr_nelbo)

        if self.run_validation:
            va_nll, va_nelbo = self._compute_negative_log_likelihood(
                self.dataloader.get_validation_set())
            best_val_nll, rerun_test = self._check_val_nll(va_nll)
            self._log('v', va_nll, va_nelbo)
            if self.display_best_val:
                _pretty_print(self.fh, 'best vNLL', best_val_nll)

        if self.run_test:
            if rerun_test:
                te_nll, te_nelbo = self._compute_negative_log_likelihood(
                    self.dataloader.get_test_set())
                self.test_nll, self.test_nelbo = te_nll, te_nelbo
            else:
                te_nll, te_nelbo = self.test_nll, self.test_nelbo
            self._log('t', te_nll, te_nelbo)

        if self.display_wait:
            _pretty_print(self.fh, 'waited', self.wait)

        if self.display_epoch:
            _pretty_print(self.fh, 'epoch', epoch)

        if self.end_line:
            _print(self.fh, '\n')

        if self.run_validation and rerun_test and self.save_model_path is not None:
            self.model_saver.on_epoch_end(epoch, logs)

        if self.wait >= self.lr_patience and self.step_decay < 1.0:
            lr = float(self.model.keras_model.optimizer.lr.get_value())
            new_lr = lr * self.step_decay
            self.model.keras_model.optimizer.lr.set_value(new_lr)
            _print(self.fh,
                   ("Decaying learning rate from {:f} to {:f}\n"
                    .format(float(lr), float(new_lr)))
            )
            self.lr_patience += (self.patience - self.lr_patience)/2

        if self.wait >= self.patience:
            _print(self.fh, "Exiting early\n")
            self.model.stop_training = True

class LossLog(Callback):
    """Prints the loss log"""
    def __init__(self, fh=sys.stdout, end_line=False):
        self.fh = fh
        self.end_line = end_line

    def on_epoch_end(self, epoch, logs={}):
        losses = logs['losses']
        loss_names = self.model.loss_names
        losses = ["{:9.3e}".format(float(loss)) for loss in tuplify(losses)]

        for name, loss in zip(loss_names, losses[-len(loss_names):]):
            if name != 'ignore':
                _print(self.fh, '{0:s}: {1:s} ;'.format(name, loss))

        if self.end_line:
            _print(self.fh, '\n')

class ModelSaver(Callback):
    """Saves the models"""
    def __init__(self, filepath, fh=sys.stdout):
        self.fh = fh
        folder = os.path.dirname(filepath)
        if not os.path.exists(folder):
            print "Creating directory:", folder
            os.makedirs(folder)

        print "Saving model to {:s}".format(filepath)
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        self.model.keras_model.save_weights(self.filepath, overwrite=True)
