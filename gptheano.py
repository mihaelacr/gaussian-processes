from theano import tensor as T
import theano
import numpy as np

# TODO: make a working example of this and compare it with the numpy one

class GaussianProcess(object):

  """ Initialize the object
    noise: what noise level do you expect in your observations (under Gaussian assumption)
    covFunction an object of type CovarianceFunction
  """
  def __init__(self, covFunction, noise=0):
    self.covFunction = covFunction
    self.observedX = None
    self.observedY = None
    self.noise = noise

  # GPS have no state, to when you fit you do not
  # so far let us assume that these are theano tensors or shared variables
  def fit(self, x, y):
    if self.observedX is None:
      assert self.observedY is None
      self.observedX = x
      self.observedY = y
    else:
      self.observedX = T.concatenate([self.observedX, x], axis=0)
      self.observedY = T.concatenate([self.observedY, y], axis=0)

  def getDataCovMatrix(self):
    return self.covFunction.covarianceMatrix(self.observedX)

  def predict(self, x):
    # Take the noise into account
    # in the book they to chelesky here
    K_observed_observed = self.getDataCovMatrix() + self.noise ** 2
    # TODO: check how this works
    inv_K_observed_observed = T.nlinalg.matrix_inverse(K_observed_observed)
    # again think of elementwise operations

    dataInstances = self.observedX.shape[0]
    repeatedX = T.extra_ops.repeat(x, dataInstances, axis=0)

    K_predict_observed = self.covFunction.apply(repeatedX, self.observedX)
    K_observed_predict = self.covFunction.apply(self.observedX.T, repeatedX.T)
    K_predict_predict  = self.covFunction.apply(x, x)

    mean = dot([K_predict_observed, inv_K_observed_observed, self.observedY])
    mean = mean[0]

    covariance = K_predict_predict - dot([K_predict_observed, inv_K_observed_observed, K_observed_predict])
    # do we still need this?
    covariance = covariance.ravel()
    covariance = covariance[0]

class CovarianceFunction(object):

  def __init__(self, hyperparmeters=None):
    self.hyperparmeters  = hyperparmeters

  def apply(self, x1, x2):
    raise NotImplementedError("cannot call apply on CovarianceFunction, use it only for subclasses")

  # Builds a covariance matrix using the covariance function and the data given (xs)
  def covarianceMatrix(self, xs):
    size = xs.shape[0]
    xmat = T.extra_ops.repeat(xs, xs.shape[0], axis=0).reshape((size, size))
    return self.apply(xmat.T, xmat)

""" Example covariance functions"""

# Stationary covariance function
class SquaredExponential(CovarianceFunction):

  def apply(self, x1, x2):
    return T.exp(-(x1- x2)**2)

# Stationary covariance function
# Incorporate hyperparams
class CubicExponential(CovarianceFunction):

  def apply(self, x1, x2):
    return T.exp(- T.abs(x1- x2)**3)


def dot(mats):
  return reduce(T.dot, mats)