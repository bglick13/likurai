import pymc3 as pm
from theano.tensor.nnet import relu, softmax
from numpy.random import randn

from .layer import BayesianDenseLayer