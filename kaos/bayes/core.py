from keras.layers import Input
from keras.models import Model
from keras.callbacks import CallbackList
from kaos.utils import tuplify, listify
from kaos.nputils import log_sum_exp
import itertools
import numpy as np


class BayesNet(object):
    """Abstract class for all Bayesian networks.

    The underlying assumption of the variational framework is that the
    relationship between variables can be described by a Bayesian network.
    Objects that inherit BayesNet will define the generative network, the
    inference network, and the loss function (negative variational
    lowerbound).
    """
    def __init__(self, **kwargs):
        """Initialize BayesNet
        """
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    def _define_io_loss(self):
        """Subclass must define the Keras model inputs, outputs, and loss."""
        raise NotImplementedError

    def log_importance_likelihood(self):
        """Subclass must define how the log imporatnce likelihood is computed"""
        raise NotImplementedError

    def _standardize_io_loss(self, inputs, outputs, losses, hidden_outputs=[]):
        """Establishes a standard for input-output-loss lists."""
        inputs = listify(inputs, recursive=True)
        outputs = listify(outputs, recursive=True)
        losses = listify(losses, recursive=True)
        hidden_outputs = listify(hidden_outputs, recursive=True)

        if len(outputs) != len(losses):
            raise Exception("The number of outputs ({}) does not "
                            .format(len(outputs)) +
                            "match the number of losses defined ({})."
                            .format(len(losses)))
        return inputs, outputs, losses, hidden_outputs

    def _zip_io_losses(self, *io_losses):
        """Collect multiple input-output-loss lists.

        Args:
            io_losses (tuple): A tuple of lists of inputs-outputs-losses.

        Returns:
            list: A single list of inputs-outputs-losses.
        """
        return self._standardize_io_loss(*zip(*io_losses))

    def log_likelihood(self, data, n_samples):
        """Computes the log likelihood using the log-sum-exp trick.

        Args:
            data_input (list, np.ndarray): Input provided by dataloader.
            data_output: Output provided by data loader.
            n_samples (int): Number of latent samples for approximating the LL.

        Returns:
            ll (np.ndarray): approx. log likelihood of each data sample.
            elbo (np.ndarray): approx. ELBO of each data sample.
        """
        inputs, outputs = data
        log_imp = self.log_importance_likelihood(inputs, outputs, n_samples)
        ll = log_sum_exp(log_imp, axis=-1) - np.log(n_samples)
        elbo = log_imp.mean(axis=-1)
        return ll, elbo

    def nll(self, data, n_samples=50):
        """Convenience function for computing negative log likelihood"""
        ll, elbo = self.log_likelihood(data, n_samples=n_samples)
        nll = -np.mean(ll)
        nelbo = -np.mean(elbo)
        return nll, nelbo

    def compile(self, optimizer, loss_weights=None,
                compute_log_likelihood=False, verbose=0):
        """Compiles the Keras model that BayesNet uses internally.

        Args:
            optimizer (str, Optimizer): Optimizer used for parameter updates.
            loss_weights (list): List of loss weights to be applied to model.
            compute_log_likelihood (bool): Checks whether the LL should be
                defined.
        """
        self._compute_log_likelihood = compute_log_likelihood
        if verbose > 0: print "[BayesNet] Defining graph..."
        inputs, outputs, losses, hidden_outputs = self._define_io_loss()
        # preserve direct access to variables for debugging purposes
        self.debug_info = {'inputs': inputs,
                           'outputs': outputs,
                           'losses': losses,
                           'hidden_outputs': hidden_outputs}
        self.loss_names = [loss.__name__ for loss in losses]
        self.keras_model = Model(inputs, outputs)

        # Hidden outputs trick
        if len(hidden_outputs) > 0:
            if verbose > 0: print "[BayesNet] Defining hidden model..."
            self.hidden_model = Model(inputs, outputs + hidden_outputs)
            self.keras_model.layers = self.hidden_model.layers

        if verbose > 0: print "[BayesNet] Compiling model..."
        self.keras_model.compile(optimizer=optimizer,
                                 loss=losses,
                                 loss_weights=loss_weights)
        if verbose > 0: print "[BayesNet] Compilation complete"

    @staticmethod
    def _get_iterations(nb_iter, nb_epoch, iter_per_epoch):
        """Process the inputs for iter, epoch, and iter/epoch."""
        if nb_epoch is None and nb_iter is None:
            raise Exception("Must define either nb_epoch or nb_iter.")
        if nb_epoch is not None and nb_iter is not None:
            raise Exception("Do not specify nb_iter and nb_epoch "
                            "simultaneously.")
        if nb_epoch is not None and iter_per_epoch is None:
            raise Exception("Must specify iter_per_epoch when using nb_epoch.")
        if nb_iter is None:
            nb_iter = nb_epoch * iter_per_epoch
        if iter_per_epoch is None:
            iter_per_epoch = nb_iter
        return nb_iter, iter_per_epoch

    def fit(self, dataloader, nb_iter=None, nb_epoch=None, iter_per_epoch=None,
            callbacks=[]):
        """Trains the underlying Keras model.

        Args:
            dataloader (StandardDataLoader): Manages the loading of data to
                model.
            nb_iter (int): The number of iterations to train the model.
            nb_epoch (int): The number of epochs to train the model.
            iter_per_epoch (int): Defines the number of iterations per epoch.
            callbacks (list): List of Keras callbacks to run during training.
        """
        nb_iter, iter_per_epoch = self._get_iterations(
            nb_iter, nb_epoch, iter_per_epoch)
        callbacks = CallbackList(callbacks)
        callbacks._set_model(self)
        callbacks.on_train_begin()

        try:
            epoch = 0
            self.stop_training = False
            for i in xrange(nb_iter):
                # Begin epoch
                if i % iter_per_epoch == 0:
                    callbacks.on_epoch_begin(epoch)
                # Execution
                callbacks.on_batch_begin(i)
                losses = self.keras_model.train_on_batch(
                    *dataloader.get_training_batch())
                callbacks.on_batch_end(i)
                # End epoch
                if (i + 1) % iter_per_epoch == 0:
                    callbacks.on_epoch_end(epoch, logs={'losses': losses})
                    epoch += 1
                if self.stop_training:
                    break
        except KeyboardInterrupt:
            print "\n[BayesNet] Abort: KeyboardInterrupt"
            raise

        callbacks.on_train_end()
