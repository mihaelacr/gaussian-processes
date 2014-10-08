from theano import tensor as T
import theano
import numpy as np

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

# let's just test that what we have done so far is the same as the numpy version
def main():
  exp = SquaredExponential()

  xs = T.dvector('xs')
  mat = exp.covarianceMatrix(xs)

  fun = theano.function(inputs=[xs], outputs=mat, updates={})

  var = np.array([1, 2])

  print fun(var)


  # make a function that just runs the theano code


if __name__ == '__main__':
  main()