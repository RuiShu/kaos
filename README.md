# TensorBayes
Variational neural networks for Keras

# Install

Install from pip

      pip install tensorbayes


Install from source (development version)

      git clone https://www.github.com/RuiShu/kaos
      cd kaos
      python setup.py install


# Philosophy
The main idea behind Kaos is to separate the Bayesian network (i.e. the set of
conditional distributions) from the neural networks (i.e. the function
approximators that underlie the conditional distributions).

Taking a standard VAE, for example, the Bayesian network is simply:
```
p(x, z) = p(z)p(x | z)
p(z) = Unit Gaussian
p(x | z) = Conditional Gaussian
```
The variational inference procedure relies on a variational approximation of the posterior and is described as:
```
q(z | x) = Conditional Gaussian
```
And the variational lowerbound is simply:
```
p(x) >= E[ln p(x, z) - ln q(z | x)],
```
where, the expectation is over `z ~ q(z | x)`.

Notice that the definition of the Bayesian network, the inference network, and
the objective function does not depend on the choice of neural networks. Indeed,
the neural networks only play a role in defining how the distributions are
actually computed. For example,
```
q(z | x) = N(z | mu(x), var(x)),
```
where `mu` and `var` are the neural networks that parameterize the distribution.
This separation means that the Bayesian network can be defined first without
needing to worry about the neural networks that parameterize it. Keeping the two
separate produces clean, readable code.

In particular, we recommend the following paradigm:
```
BayesNet Model: A BayesNet model defines the desired Bayesian network and loss function
DataLoader: A data loader feeds the data
Training file: The training file defines the neural networks used to parameterize the distributions
```

We provide examples of how to do so in the `examples` folder. Thus far, standard
VAE, ladder VAEs, and auxiliary VAEs are easily implementable. Any variants of
these models are also easily implementable. Normalizing flow is also possible,
and will likely be implemented in the future.

This library uses Keras/Theano. A Tensorflow version is currently in the works.
