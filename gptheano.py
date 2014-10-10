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
  def __init__(self, covFunction, noise=0.0):
    self.covFunction = covFunction
    self.observedX = None
    self.observedY = None
    self.noise = noise

  def predict(self, x):
    return self.predictFun(x)

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

  """ Creates the theano function that will do the prediction and sets it
      as a field in the GP object.
      This is required because defining the predictTheano function in the predict
      method would imply that theano compiles that function every time we call predict,
      resulting in a substantial slow down of the code.
  """
  def _createTheanoPredictFunction(self):
    predictionVar = T.dvector("predictionVar")

    mean, covariance = self._predictTheano(predictionVar)

    predictFun = theano.function(inputs=[predictionVar],
                                 outputs=[mean, covariance],
                                 givens = self.covFunction.updateDict)

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

    mean = dot([KPredictObserved, invKObservedObserved, self.observedVarY])

    covariance = KPredictPredict - dot([KPredictObserved, invKObservedObserved, KObservedPredict])
    return mean, covariance

  """ Predicts multiple data instances."""
  def predictAll(self, xs):
    predictions = map(self.predictFun, xs)
    means = np.array([p[0] for p in predictions])
    covariances = np.array([p[1] for p in predictions])
    return means, covariances

  # note that in the Bayesian opt paper they also add a mean M for which I did not account yet
  # you have to integrate the mean into the code as well
  def loglikelihood(self):
    covarianceData = self.covFunction.covarianceMatrix(self.observedVarX) + self.noise ** 2
    # determinant
    return T.log(1./ np.sqrt(2 * np.pi)) - 1./2 * dot([self.observedVarX.T, covarianceData, self.observedVarX])

  # you can actually pass the theano functions to scipy optimize, let's try that
  def optimizehyperparams(self):
    hyperparmeters = self.covFunction.hyperparmeters
    likelihood = self.loglikelihood()
    print "likelihood.ndim"
    print likelihood.ndim
    grad = T.grad(likelihood, hyperparmeters)
    # TODO: add the bounds that were in the paper

    hypers = optimize.fmin_l_bfgs_b(likelihood, hyperparmeters, grad, args=(), bounds=None, disp=0)

    # but wait, is it OK if these are numbers and not numpy vars?
    print hypers

    # set the hyperparameters to the optimized ones

class CovarianceFunction(object):

  def covarianceMatrix(self, x1, x2=None):
    raise NotImplementedError("cannot call covarianceMatrix on CovarianceFunction, only on subclasses")


""" Example covariance functions"""
# Stationary covariance function
class SquaredExponential(CovarianceFunction):

  def __init__(self):
    self.hyperparmeters = []
    self.updateDict = {}

  def covarianceMatrix(self, x1Mat, x2Mat=None):
    return T.exp(- distanceSquared(x1Mat, x2Mat))

  def applyVecMat(self, vec, mat):
    return T.exp(-T.sum((vec - mat) ** 2, axis=1))

  def applyVecVec(self, vec1, vec2):
    return T.exp(-T.sum((vec1 - vec2) ** 2))

class ARDSquareExponential(CovarianceFunction):

  def __init__(self, inputSize, hyperparmeterValues=None):
    self.hyperparmeterValues = hyperparmeterValues
    if hyperparmeterValues is None:
      self.l0Val = 1.0
      self.lsVal = np.ones(inputSize, dtype='float64')
      self.hyperparmeterValues = np.ones(inputSize + 1, dtype='float64')
    else:
      self.l0Val = hyperparmeterValues[0]
      self.lsVal = hyperparmeterValues[1:]

    self.l0 = T.dscalar('l0')
    self.ls = T.dvector('ls')
    self.hyperparmeters = [self.l0, self.ls] # we need this for the gradients

    self.updateDict = {
       self.l0 : self.l0Val,
       self.ls:  self.lsVal
       }

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
