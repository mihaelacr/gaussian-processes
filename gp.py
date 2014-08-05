""" This is my initial implementation of gaussian processes. Made for educational purposes.
Use at your own risk."""

import numpy as np
from scipy.stats import norm as normal


# TODO: learn hyperparmeters using maximum likelihood
# that seems to be tricky because it might depend from cov function to cov function
# but you can use some basic optimization techniques

# TODO: do something with the means of the y because we assume to mean to be 0

class GaussianProcess(object):

  # covFunction is a object of type CovarianceFunction
  def __init__(self, covFunction, noise=0):
    self.covFunction = covFunction
    self.observedX = None
    self.observedY = None

  # GPS have no state, to when you fit you do not
  def fit(self, x, y):
    if self.observedX is None:
      assert self.observedY is None
      self.observedX = x
      self.observedY = y
    else:
      self.observedX = np.concatenate(self.observedX, x)
      self.observedY = np.concatenate(self.observedY, y)

  def getDataCovMatrix(self):
    return self.covFunction.covarianceMatrix(self.observedX)


  # TODO: cache all this in case the data does not change and you have to compute all the matrices again
  # for a second prediction
  def predict(self, x):
    K_observed_observed = self.getDataCovMatrix()
    inv_K_observed_observed = np.linalg.inv(K_observed_observed)
    K_predict_observed = np.array([self.covFunction.apply(x, x_obs) for x_obs in self.observedX]).reshape(1, len(self.observedX))
    K_observed_predict = np.array([self.covFunction.apply(x_obs, x) for x_obs in self.observedX]).reshape(len(self.observedX), 1)
    K_predict_predict = self.covFunction.apply(x, x)

    mean = dot([K_predict_observed, inv_K_observed_observed, self.observedY])
    mean = mean[0]

    covariance = K_predict_predict - dot([K_predict_observed, inv_K_observed_observed, K_observed_predict])
    covariance = covariance.ravel()
    covariance = covariance[0]

    y = normal(loc=mean, scale=covariance).pdf(x)
    return y



class CovarianceFunction(object):

  def __init__(self, hyperparmeters=None):
    self.hyperparmeters  = hyperparmeters

  def apply(self, x1, x2):
    pass

  # Builds a covariance matrix using the covariance function and the data given (xs)
  def covarianceMatrix(self, xs):
    return np.array([self.apply(x1, x2) for x1 in xs for x2 in xs]).reshape((len(xs), len(xs)))


""" Example covariance functions"""

# TODO: you can vectorize these covariance functions by doing an elementwise product of the vector elements

# Stationary covariance function
class SquaredExponential(CovarianceFunction):


  def apply(self, x1, x2):
    return np.exp(-(x1- x2)**2)

def dot(mats):
  return reduce(np.dot, mats)