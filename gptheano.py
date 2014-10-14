from theano import tensor as T
import theano
# Use the sandbox version if theano is before and including 0.6.0
if theano.__version__ > "0.6.0":
  import theano.tensor.nlinalg as nl
else:
  import theano.sandbox.linalg as nl

import numpy as np
from scipy import optimize

"""Even though theano supports running code on the gpu, for this module this is not needed and it will not exhibit any advantages"""

class GaussianProcess(object):

  """ Initialize the object
    noise: what noise level do you expect in your observations (under Gaussian assumption)
          note that noise is also a hyperparmeter which can be optimized using max likelihood
    covFunction an object of type CovarianceFunction
  """
  def __init__(self, covFunction, mean=0.0, noise=0.0):
    self.covFunction = covFunction
    self.mean = mean
    self.observedX = None
    self.observedY = None
    self.noise = noise

  def predict(self, x):
    hyper = self.covFunction.hyperparameterValues
    return self.covFunction.callFunctiononHyperparamesWithOtherParamFirst(self.predictFun, x, hyper)

  # TODO: work with being able to add data incrementally
  def fit(self, x, y):
    print "fitting data"
    print "observations shape", x.shape

    if len(x.shape) == 1:
      resX = x.reshape((x.shape[0], 1))
      print "resizing the input to be a matrix instead of a vector"
      print "previous shape " + str(x.shape) + " current shape " + str(resX.shape)

    # Observed data as numpy variables
    self.observedX = x
    self.observedY = y

    # Symbolic variables that contain the data to fit into the GP
    self.observedVarX = T.as_tensor_variable(x, name='varX')
    self.observedVarY = T.as_tensor_variable(y, name='varY')

    # The symbolic variable for prediction
    self.predictionVar = T.dvector("predictionVar")
    self._createTheanoPredictFunction()
    self._createTheanoLogFunction()
    self._createTheanoLogGradFunction()

  """ Creates the theano function that will do the prediction and sets it
      as a field in the GP object.
      This is required because defining the predictTheano function in the predict
      method would imply that theano compiles that function every time we call predict,
      resulting in a substantial slow down of the code.
  """
  def _createTheanoPredictFunction(self):
    predictionVar = T.dvector("predictionVar")

    mean, covariance = self._predictTheano(predictionVar)

    inputs = [predictionVar] + self.covFunction.hyperparameters
    predictFun = theano.function(inputs=inputs,
                                 outputs=[mean, covariance])

    self.predictFun = predictFun

  """ The theano code which contains the prediction logic."""
  def _predictTheano(self, x):
    KObservedObserved =  self.covFunction.covarianceMatrix(self.observedVarX) + self.noise ** 2

    # TODO: Move to cholesky when possible
    # after theano implemented solve_triangular
    invKObservedObserved = nl.matrix_inverse(KObservedObserved)

    KPredictObserved = self.covFunction.applyVecMat(x, self.observedVarX)
    KObservedPredict = self.covFunction.applyVecMat(self.observedVarX, x)
    KPredictPredict  = self.covFunction.applyVecVec(x, x)

    mean = self.mean + dot([KPredictObserved, invKObservedObserved, self.observedVarY - self.mean])

    covariance = KPredictPredict - dot([KPredictObserved, invKObservedObserved, KObservedPredict])
    return mean, covariance


  """ Predicts multiple data instances."""
  def predictAll(self, xs):
    predictions = map(self.predict, xs)
    means = np.array([p[0] for p in predictions])
    covariances = np.array([p[1] for p in predictions])
    return means, covariances

  def _theanolog(self):
    covarianceMatrix = self.covFunction.covarianceMatrix(self.observedVarX) + self.noise ** 2
    invKObservedObserved = nl.matrix_inverse(covarianceMatrix)

    yVarMean = self.observedVarY - self.mean
    loglike = T.log(1./ T.sqrt(2 * np.pi * nl.det(covarianceMatrix))) - 1./2 * dot([yVarMean.T, invKObservedObserved, yVarMean])
    return loglike

  def _createTheanoLogFunction(self):
    loglike = self._theanolog()
    logFun = theano.function(inputs=self.covFunction.hyperparameters,
                                 outputs=[loglike])

    self.logFun = logFun

  def _createTheanoLogGradFunction(self):
    loglike = self._theanolog()
    gradLike = T.grad(loglike, self.covFunction.hyperparameters)

    logGradFun = theano.function(inputs=self.covFunction.hyperparameters,
                                 outputs=gradLike)

    self.logGradFun = logGradFun

  """ Get loglikelihood for the hyperparams which are a numpy because we have to
  use this for scipy optimize"""
  def loglikelihood(self, hyperparameterValues):
    return self.covFunction.callFunctiononHyperparames(self.logFun, hyperparameterValues)

  def loglikilhoodgrad(self, hyperparameterValues):
    return np.array(self.covFunction.callFunctiononHyperparames(self.logGradFun, hyperparameterValues))

  def optimizehyperparams(self):
    init = self.covFunction.hyperparameterValues

    b = [(-1000, 1000), (-1000, 1000)] # TODO: make this proper

    hypers = optimize.fmin_l_bfgs_b(self.loglikelihood, x0=init,
                                     fprime=self.loglikilhoodgrad,
                                     args=(), bounds=b, disp=0)
    hypers = hypers[0] # optimize also returns some data about the procedure, nore that
    print hypers
    self.covFunction.hyperparameterValues = hypers
    return hypers


class CovarianceFunction(object):

  def covarianceMatrix(self, x1, x2=None):
    raise NotImplementedError("cannot call covarianceMatrix on CovarianceFunction, only on subclasses")


""" Example covariance functions"""
# Stationary covariance function
class SquaredExponential(CovarianceFunction):

  def __init__(self):
    self.hyperparameters = []
    self.updateDict = {}

  def covarianceMatrix(self, x1Mat, x2Mat=None):
    return T.exp(- distanceSquared(x1Mat, x2Mat))

  def applyVecMat(self, vec, mat):
    return T.exp(-T.sum((vec - mat) ** 2, axis=1))

  def applyVecVec(self, vec1, vec2):
    return T.exp(-T.sum((vec1 - vec2) ** 2))

# Stationary covariance function
class ARDSquareExponential(CovarianceFunction):

  def __init__(self, inputSize, hyperparameterValues=None):
    if hyperparameterValues is None:
      self.hyperparameterValues = np.ones(inputSize + 1, dtype='float64')
    else:
      self.hyperparameterValues = hyperparameterValues

    self.l0 = T.dscalar('l0')
    self.ls = T.dvector('ls')
    self.hyperparameters = [self.l0, self.ls] # we need this for the gradients


  def callFunctiononHyperparames(self, func, hyperparameterValues):
    return func(hyperparameterValues[0], hyperparameterValues[1:])

  def callFunctiononHyperparamesWithOtherParamFirst(self, func, first, hyperparameterValues):
    return func(first, hyperparameterValues[0], hyperparameterValues[1:])

  def covarianceMatrix(self, x1Mat, x2Mat=None):
    return self.l0 * T.exp(- distanceSquared(x1Mat, x2Mat, self.ls))

  def applyVecMat(self, vec, mat):
    vec = vec / self.ls
    mat = mat / self.ls # TODO: ensure this gets broadcasted properly

    return self.l0 * T.exp(-T.sum((vec - mat) ** 2, axis=1))

  def applyVecVec(self, vec1, vec2):
    vec1 = vec1 / self.ls
    vec2 = vec2 / self.ls

    return self.l0 * T.exp(-T.sum((vec1 - vec2) ** 2))



"""
  Computes the square of the euclidean distance in a vectorized fashion, to avoid for loops
  What this function does is compute the squared euclidean distance between all row vectors of matrix xs
  and puts the result in a matrix. The equivalent python code would be:
  for i, v in enumerate(mat):
    for j, u in enumerate(mat):
      a[i, j ] = euclideanSquared(u, v)

  Taken from: https://github.com/JasperSnoek/spearmint/gp.py
 """
def distanceSquared(x1, x2=None, ls=None):
  if x2 == None:
    x2 = x1

  x1 = x1 / ls
  x2 = x2 / ls

  m = -(T.dot(x1, 2*x2.T)
          - T.sum(x1*x1, axis=1)[:,np.newaxis]
          - T.sum(x2*x2, axis=1)[:,np.newaxis].T)
  res = m * (m > 0.0)
  assert res.ndim == 2, "covariance matrix does not have 2 dimensions"
  return res

def dot(mats):
  return reduce(T.dot, mats)

