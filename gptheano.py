from theano import tesnor as T

class CovarianceFunction(object):

  def __init__(self, hyperparmeters=None):
    self.hyperparmeters  = hyperparmeters

  def apply(self, x1, x2):
    raise NotImplementedError("cannot call apply on CovarianceFunction, use it only for subclasses")

  # Builds a covariance matrix using the covariance function and the data given (xs)
  def covarianceMatrix(self, xs):
    xmat = T.extra_ops.repeat(xs, xs.shape[0], axis=0)
    return self.apply(xmat, xmat.T)


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