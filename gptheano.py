from theano import tensor as T
import theano.tensor.nlinalg as nl
import theano
import numpy as np

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

  def predict(self, x):
    return self.predictFun(x)

  def fit(self, x, y):
    print "fitting data"
    print "x", x.shape
    print "y", y

    if len(x.shape) == 1:
      resX = x.reshape((x.shape[0], 1))
      print "resizing the input to be a matrix instead of a vector"
      print "previous shape " + str(x.shape) + " current shape " + str(resX.shape)

    self.observedX = x
    self.observedY = y
    self.observedVarX = T.as_tensor_variable(x, name='varX')
    self.observedVarY = T.as_tensor_variable(y, name='varY')
    self.predictionVar = T.dvector("predictionVar")
    self._createTheanoPredictFunction()

  def _createTheanoPredictFunction(self):
    predictionVar = T.dvector("predictionVar")

    mean, covariance = self._predictTheano(predictionVar)

    predictFun = theano.function(inputs=[predictionVar],
                                 outputs=[mean, covariance])
    self.predictFun = predictFun


  def _predictTheano(self, x):
    KObservedObserved =  self.covFunction.covarianceMatrix(self.observedVarX) + self.noise ** 2

    # TODO: check how this works, move to cholesky if possible (doesn't look like it, theano does not have solve_triangular yet)
    invKObservedObserved = nl.matrix_inverse(KObservedObserved)

    KPredictObserved = self.covFunction.applyVecMat(x, self.observedVarX)
    KObservedPredict = self.covFunction.applyVecMat(self.observedVarX, x)
    KPredictPredict  = self.covFunction.applyVecVec(x, x)

    mean = dot([KPredictObserved, invKObservedObserved, self.observedVarY])

    covariance = KPredictPredict - dot([KPredictObserved, invKObservedObserved, KObservedPredict])
    return mean, covariance

  # you can memoize the covariance matrix to mkae this faster (and the inverse, tht is probably the slow part)
  def predictAll(self, xs):
    predictionVar = T.dvector("predictionVar")

    mean, covariance = self._predictTheano(predictionVar)

    predictFun = theano.function(inputs=[predictionVar],
                                 outputs=[mean, covariance])

    # return mean, covariance
    predictions = map(predictFun, xs)
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

  def applyVecMat(self, vec, mat):
    return T.exp(-T.sum((vec - mat) ** 2, axis=1))

  def applyVecVec(self, vec1, vec2):
    return T.exp(-T.sum((vec1 - vec2) ** 2))

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
