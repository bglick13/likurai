import pymc3 as pm
import theano.tensor.nnet as nn
from theano.tensor import concatenate
from theano.tensor.nnet import relu, softmax, sigmoid
from numpy.random import randn
from functools import partial
from .layer import BayesianDenseLayer, Layer


def retrieve_distribution(dist):
    return getattr(pm, dist)


def retrieve_activation(act):
    return getattr(nn, act)
