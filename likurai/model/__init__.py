from .model import Model
from ..layer import BayesianDenseLayer, Layer
import numpy as np
import pymc3 as pm
from theano import shared
import theano.tensor as tt