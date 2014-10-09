from theano import tensor as T
from theano.sandbox import linalg
import theano
import numpy as np

# TODO: make a working example of this and compare it with the numpy one

class GaussianProcess(object):

  """ Initialize the object
    noise: what noise level do you expect in your observations (under Gaussian assumption)
    covFunction an object of type CovarianceFunction
  """
  def __init__(self, covFunction, noise=0.0):
    self.covFunction = covFunction
    self.observedX = None
    self.observedY = None
    self.noise = noise

    self.observedVarX = T.dmatrix("observedVarX")
    self.observedVarY = T.dvector("observedVarY")

  def predict(self, x):
    print "x", x
    predictionVar = T.dvector("predictionVar")

    # predict using theano code
    mean, covariance = self.predictTheano(predictionVar)

    predictFun = theano.function(inputs=[],
      outputs=[mean, covariance],
      givens = {
        self.observedVarX: self.observedX,
        self.observedVarY: self.observedY,
        predictionVar: x,
      })

    mean, covariance =  predictFun()
    print "mean", mean
    print "covariance", covariance
    return mean, covariance

  def fit(self, x, y):
    print "fitting data"
    print "x", x.shape
    print "y", y

    if self.observedX is None:
      assert self.observedY is None
      if len(x.shape) == 1:
        resX = x.reshape((x.shape[0], 1))
        print "resizing the input to be a matrix instead of a vector"
        print "previous shape " + str(x.shape) + " current shape " + str(resX.shape)

      self.observedX = x
      self.observedY = y
    else:
      # the axis might have to be specified here
      self.observedX = np.concatenate(self.observedX, x)
      self.observedY = np.concatenate(self.observedY, y)

  def getDataCovMatrix(self):
    return self.covFunction.covarianceMatrix(self.observedVarX)

  def predictTheano(self, x):
    # Take the noise into account
    # in the book they to chelesky here
    K_observed_observed = self.getDataCovMatrix() + self.noise ** 2

    # TODO: check how this works, move to cholesky if possible
    inv_K_observed_observed = linalg.matrix_inverse(K_observed_observed)

    nrDataInstances =  self.observedVarX.shape[0]
    lenX = x.shape[0]

    repeatedX = T.extra_ops.repeat(x, nrDataInstances, axis=0).reshape((nrDataInstances, lenX))

    # Too much computation, try to reduce it
    K_predict_observed = self.covFunction.covarianceMatrix(repeatedX, self.observedVarX)[:, 0]
    K_observed_predict = self.covFunction.covarianceMatrix(self.observedVarX, repeatedX)[0, :]
    K_predict_predict  = self.covFunction.apply(x, x) # this has to change, you can keep the apply version of the code

    mean = dot([K_predict_observed, inv_K_observed_observed, self.observedVarY])

    covariance = K_predict_predict - dot([K_predict_observed, inv_K_observed_observed, K_observed_predict])
    return mean, covariance

  # you can memoize the covariance matrix to mkae this faster (and the inverse, tht is probably the slow part)
  def predictAll(self, xs):
    predictions = map(self.predict, xs)
    means = np.array([p[0] for p in predictions])
    covariances = np.array([p[1] for p in predictions])
    return means, covariances


class CovarianceFunction(object):

  def __init__(self, hyperparmeters=None):
    self.hyperparmeters  = hyperparmeters

  def covarianceMatrix(self, x1, x2=None):
    raise NotImplementedError("cannot call covarianceMatrix on CovarianceFunction, only on subclasses")

""" Example covariance functions"""
# Stationary covariance function
class SquaredExponential(CovarianceFunction):

  def covarianceMatrix(self, x1Mat, x2Mat=None):
    return T.exp(- distanceSquared(x1Mat, x2Mat))

  def apply(self, x1, x2):
    return T.exp(-T.sum((x1 - x2) ** 2))

"""
  Computes the square of the euclidean distance in a vectorized fashion, to avoid for loops
  What this function does is compute the squared euclidean distance between all row vectors of matrix xs
  and puts the result in a matrix. The equivalent python code would be:
  for i, v in enumerate(mat):
    for j, u in enumerate(mat):
      a[i, j ] = euclideanSquared(u, v)

  Taken from: https://github.com/JasperSnoek/spearmint/gp.py
 """
def distanceSquared(x1, x2=None):
  if x2 == None:
    x2 = x1

  m = -(T.dot(x1, 2*x2.T)
          - T.sum(x1*x1, axis=1)[:,np.newaxis]
          - T.sum(x2*x2, axis=1)[:,np.newaxis].T)
  res = m * (m > 0.0)
  assert res.ndim == 2, "covariance matrix does not have 2 dimensions"
  return res

def dot(mats):
  return reduce(T.dot, mats)
