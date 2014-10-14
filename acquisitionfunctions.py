import theano
import numpy as np
from theano import tensor as T
import gptheano
import sobol
from scipy import optimize

# TODO: MCMC samples to get the integrated acquisition functions
# so far we just use the optimized hyperparameters of the gp
NR_CANDIDATES = 10


def AcquisitionExpectedImprovement(object):

  def __init__(self, gp):
    self.gp = gp
    self.__createExpImpovTheanoFunction()
    self.__createExpImpovTheanoFunctionGrad()

  def aExpectedImprovementTheano(self, x):
    mean, covariance = self.gp._predictTheano(x)
    currentBest = T.min(self.gp.observedVarY)
    gammaX = gamma(x, self.gp, currentBest)
    return covariance * (gammaX * cdf(gammaX) + pdf(gamma, 0.0, 1.0))

  def __createExpImpovTheanoFunction(self):
    x = T.dvector("x")
    improvement = self.aExpectedImprovement(x)

    aExpImprovFunc = theano.function(inputs=[x],
                                     outputs=[improvement])
    self.aExpImprovFunc = aExpImprovFunc

  def __createExpImpovTheanoFunctionGrad(self):
    x = T.dvector("x")
    improvement = self.aExpectedImprovement(x)
    grad = T.grad(improvement, x)

    aExpImprovFuncGrad = theano.function(inputs=[x],
                                     outputs=[grad])

    self.aExpImprovFuncGrad = aExpImprovFuncGrad

  def expectedImprovement(self, x):
    return self.aExpImprovFunc(x)

  def expectedImprovementGrad(self, x):
    return self.acquisitionFuncGrad(x)

# Expected improvement
# Note that the currentBest can be obtained from the GP
# because it is the biggest / smallest y in the gp values
# this allows you to remove the currentBestParameter
# this has to do with theano because we want to differentiate it
# maybe you should use the predict with theano variables not this one


""" Maximize expected improvement"""

# This gives you the next point to evaluate
# now you call your function f on this
# and you come back with a result
# you have to integrate that into the gaussian process
# in case you allow sampling then here you have to change with the integrated
# acquisition function
# that should not be too hard
def nextPointToEvaluate(aei):
  candidates = sobol.i4_sobol_generate(aei.gp.dataDimnension, NR_CANDIDATES, 100)
  # TODO: add the bounds for the optimization
  bestCandidate = None
  bestFunVal = 0
  for candidate in candidates:
    candidate = optimize.fmin_l_bfgs_b(aei.expectedImprovement, candidate, aei.expectedImprovementGrad)
    # Here you take into account if you do a min or a max
    # So far, I have gone for a max
    if aei.expectedImprovement(candidate) > bestFunVal:
      bestCandidate = candidate
      bestFunVal = aei.expectedImprovement(candidate)

  return bestCandidate


def gamma(x, gp, currentBest):
  mean, covariance = gp.predict(x)
  return (currentBest - mean) / covariance

def cdf(x, miu=0.0, variance=1.0):
  return 1.0/2 *  (1.0 + T.erf((x - miu)/ T.sqrt(2 * variance)))

def pdf(x, miu, variance):
  return 1./ (2 * np.pi* np.sqrt(variance)) * np.exp( (x - miu)** 2 / (2.0 * variance))





