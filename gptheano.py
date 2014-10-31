from theano import tensor as T
import theano
# Use the sandbox version if theano is before and including 0.6.0
if theano.__version__ > "0.6.0":
  import theano.tensor.nlinalg as nl
  import theano.tensor.slinalg as sl
else:
  import theano.sandbox.linalg as nl
  import theano.sandbox.slinalg as sl

import numpy as np
from scipy import optimize
from slice_sampling import sliceSample

"""Even though theano supports running code on the gpu, for this module this is not needed and it will not exhibit any advantages"""

class GaussianProcess(object):

  """ Initialize the object
    noise: what noise level do you expect in your observations (under Gaussian assumption)
          note that noise is also a hyperparmeter which can be optimized using max likelihood
    covFunction an object of type CovarianceFunction
  """
  def __init__(self, covFunction, mean=0.0, noise=0.01):
    self.covFunction = covFunction
    self.mean = mean
    self.observedX = None
    self.observedY = None
    self.noise = noise

    self.meanVar = T.dscalar('meanVar')
    self.noiseVar = T.dscalar('noiseVar')

    self.hyperparameters = [self.meanVar, self.noiseVar, self.covFunction.hyperparameters]

  def predict(self, x):
    return self.predictFun(x, self.covFunction.hyperparameterValues)

  # TODO: work with being able to add data incrementally
  def fit(self, x, y):
    print "fitting data"
    print "observations shape", x.shape

    if len(x.shape) == 1:
      resX = x.reshape((x.shape[0], 1))
      print "resizing the input to be a matrix instead of a vector"
      print "previous shape " + str(x.shape) + " current shape " + str(resX.shape)

    # Observed data as numpy variables
    if self.observedX is None:
      assert self.observedY is None
      self.observedX = x
      self.observedY = y
    else:
      self.observedX = np.concatenate([self.observedX, x])
      self.observedY = np.concatenate([self.observedY, y])

    # Symbolic variables that contain the data to fit into the GP
    self.observedVarX = T.as_tensor_variable(self.observedX, name='varX')
    self.observedVarY = T.as_tensor_variable(self.observedY, name='varY')

    # The symbolic variable for prediction
    self.predictionVar = T.dvector("predictionVar")
    self._createTheanoPredictFunction()
    self._createTheanoLogFunction()
    self._createTheanoLogGradFunction()
    self._createTheanoPosteriorFunction()

  """ Creates the theano function that will do the prediction and sets it
      as a field in the GP object.
      This is required because defining the predictTheano function in the predict
      method would imply that theano compiles that function every time we call predict,
      resulting in a substantial slow down of the code.
  """
  def _createTheanoPredictFunction(self):
    predictionVar = T.dvector("predictionVar")

    mean, covariance = self._predictTheano(predictionVar)

    inputs = [predictionVar, self.covFunction.hyperparameters]
    predictFun = theano.function(inputs=inputs,
                                 outputs=[mean, covariance])
    self.predictFun = predictFun

  def _createTheanoPosteriorFunction(self):
    post = self._posteriorTheano()
    posteriorFun = theano.function(inputs=self.hyperparameters,
                                   outputs=[post])

    self.posteriorFun = posteriorFun


  """ The theano code which contains the prediction logic."""
  def _predictTheano(self, x):
    KObservedObserved = self.covFunction.covarianceMatrix(self.observedVarX)
    KObservedObserved += self.noise ** 2 * T.identity_like(KObservedObserved)

    # TODO: Move to cholesky when possible
    # after theano implemented solve_triangular
    invKObservedObserved = nl.matrix_inverse(KObservedObserved)

    KPredictObserved = self.covFunction.applyVecMat(x, self.observedVarX)
    KObservedPredict = self.covFunction.applyVecMat(self.observedVarX, x)
    KPredictPredict  = self.covFunction.applyVecVec(x, x)

    # Use the mean constant not the mean var because this is not related to the optimization
    # so you use the hyperparameter values anyway for prediction
    mean = self.mean + dot([KPredictObserved, invKObservedObserved, self.observedVarY - self.mean])

    covariance = KPredictPredict - dot([KPredictObserved, invKObservedObserved, KObservedPredict])
    return mean, covariance

  def _posteriorTheano (self):
    covarianceMatrix = self.covFunction.covarianceMatrix(self.observedVarX)
    covarianceMatrix += self.noiseVar ** 2 * T.identity_like(covarianceMatrix)

    return normalPdfPropoprtional(self.observedVarY, self.meanVar, covarianceMatrix)

  # TODO: check how to change the algorithms to sample from the
  # log distribution
  # TODO: check how you sample for the different parts
  # Given the data in the gp, define a posterior over the hyperparameters
  # sample from that distribution to get a sample value for hyperparameters
  def sampleHyperparams(self, nrSamples):
    # p(phi | x, y) ~ p(x, y | phi) p(phi)
    # with a uniform prior on the hyperparameters
    # p(phi | x, y) ~ p(x, y | phi) = p(y | phi, x) p (x | phi) ~ p(y| phi, x)
    # p(phi | x, y) ~ p(y| phi, x)

    # you need to set the posterior to the probability given the hyperparams get this value
    # this can only be done using a theano function
    def distribution(phi):
      res = self.posteriorFun(phi[0], phi[1], phi[2:])[0]
      assert not np.isinf(res), "infinite likelihood for hyperparameters" + str(phi)
      return res
    # distribution = lambda phi: self.posteriorFun(phi)

    hyperparameterValues = np.zeros(len(self.covFunction.hyperparameterValues) + 2)
    hyperparameterValues[0] = self.mean
    hyperparameterValues[1] = self.noise
    hyperparameterValues[2:] = self.covFunction.hyperparameterValues
    # this should be an argument
    steps = 0.5 * np.ones(len(hyperparameterValues), dtype='float')

    return sliceSample(distribution, steps, nrSamples, 100, hyperparameterValues)


  """ Predicts multiple data instances."""
  def predictAll(self, xs):
    predictions = map(self.predict, xs)
    means = np.array([p[0] for p in predictions])
    covariances = np.array([p[1] for p in predictions])
    return means, covariances

  """ Only required for hyperparmeter optimization"""
  def _theanolog(self):
    covarianceMatrix = self.covFunction.covarianceMatrix(self.observedVarX)
    covarianceMatrix += self.noiseVar ** 2 * T.identity_like(covarianceMatrix)

    invKObservedObserved = nl.matrix_inverse(covarianceMatrix)

    yVarMean = self.observedVarY - self.meanVar

    loglike = T.log(1./ T.sqrt(2 * np.pi * nl.det(covarianceMatrix))) - 1./2 * dot([yVarMean.T, invKObservedObserved, yVarMean])
    return loglike

  """ Only required for hyperparmeter optimization"""
  def _createTheanoLogFunction(self):
    loglike = self._theanolog()
    logFun = theano.function(inputs=self.hyperparameters,
                                 outputs=[loglike])

    self.logFun = logFun

  """ Only required for hyperparmeter optimization"""
  def _createTheanoLogGradFunction(self):
    loglike = self._theanolog()
    gradLike = T.grad(loglike, self.hyperparameters)

    logGradFun = theano.function(inputs=self.hyperparameters,
                                 outputs=gradLike)

    self.logGradFun = logGradFun

  """ Only required for hyperparmeter optimization.
     Get loglikelihood for the hyperparams which are a numpy because we have to
  use this for scipy optimize"""
  def _loglikelihood(self, hyperparameters):
    mean = hyperparameters[0]
    noise = hyperparameters[1]
    covHyperparams = hyperparameters[2:]
    return self.logFun(mean, noise, covHyperparams)[0]

  def _loglikilhoodgrad(self, hyperparameters):
    mean = hyperparameters[0]
    noise = hyperparameters[1]
    covHyperparams = hyperparameters[2:]
    ret = self.logGradFun(mean, noise, covHyperparams)
    res = np.zeros(len(self.covFunction.hyperparameterValues) + 2, dtype='float')
    res[0] = ret[0]
    res[1] = ret[1]
    res[2: ] = ret[2]
    return res

  # here you also have to optimize the hypers from the mean function
  # you  might just have to admit it will just be a constant to not overcomplicate things

  # DO not use this. Prefer the sampling method because that is more stable
  # the optimization here does not work well
  def optimizehyperparams(self):
    init = np.zeros(len(self.covFunction.hyperparameterValues) + 2, dtype='float')
    init[0] = self.mean
    init[1] = self.noise
    init[2: ] = self.covFunction.hyperparameterValues

    print "init", init

    b = [(-10, 10), (0, 1)]  + [(-100, 100)] * (len(init) - 2)

    hypers = optimize.minimize(self._loglikelihood, x0=init, method='L-BFGS-B',
                                     jac=self._loglikilhoodgrad, bounds=b,
                                     args=())

    print "optimization status", hypers
    # assert hypers['success']
    hypers = hypers['x'] # optimize also returns some data about the procedure, ignore that

    # Now set the mean the the optimized hyperparameters
    self.mean = hypers[0]
    self.noise = hypers[1]
    self.covFunction.hyperparameterValues = hypers[2:]

    print "after optimization setting the hyperparmeters"
    print "the new mean is now", self.mean
    print "the new noise is now", self.noise
    print "the new covariance function parameters are now", self.covFunction.hyperparameterValues
    return hypers


class CovarianceFunction(object):

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

# Stationary covariance function
class ARDSquareExponential(CovarianceFunction):

  def __init__(self, inputSize, hyperparameterValues=None):
    if hyperparameterValues is None:
      self.hyperparameterValues = np.ones(inputSize + 1, dtype='float64')
    else:
      self.hyperparameterValues = hyperparameterValues

    self.hyperparameters = T.dvector('ardhypers')
    self.l0 = self.hyperparameters[0]
    self.ls = self.hyperparameters[1:]


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

# Does not seem to work with gradient, ignore this for now
def choleskySolver(a, res):
  l = sl.cholesky(a)
  intermiediate = sl.solve(l, res)
  return sl.solve(l.T, intermiediate)

# check the dimension of this
def normalPdfPropoprtional(x, mean, cov):
  return 1.0 / T.sqrt(nl.det(cov)) * T.exp(dot([(x - mean).T, nl.matrix_inverse(cov), x - mean]))