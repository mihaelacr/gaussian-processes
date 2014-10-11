"""The main file that tests """

import numpy as np
import gp
import gptheano

# Let's do some plotting
from matplotlib import pyplot as pl

def gpTest():

  def f(x):
      """The function to predict."""
      return x * np.sin(x)

  X = np.array([[1.], [3.], [5.], [6.], [7.], [8.]])
  # X = np.array([1., 3., 5., 6., 7., 8.])

  print X
  # Observations
  y = f(X).ravel()
  print y
  meanY = np.mean(y)

  # you have to subtract the mean to kind of make sure they have mean 0
  y = y - meanY

  print "meanY"
  print meanY

  # gaussianP = gp.GaussianProcess(covFunction=gp.SquaredExponential())
  gaussianP = gptheano.GaussianProcess(covFunction=gptheano.ARDSquareExponential(1))
  gaussianP.fit(X, y)
  res =  gaussianP.predict(np.array([0.0]))
  # res =  gaussianP.predict(0.0)
  print "predict"
  print res

  # Plot the function, the prediction and the 95% confidence interval based on
  # the MSE
  x = np.atleast_2d(np.linspace(0, 10, 100)).T
  print x.shape
  # my predict still does not work with mutiple instances but doing that is not hard
  y_pred, sigma = gaussianP.predictAll(x)

  y_pred = y_pred + meanY



  fig = pl.figure()
  pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
  pl.plot(X, f(X), 'r.', markersize=10, label=u'Observations')
  pl.plot(x, y_pred, 'b-', label=u'Prediction')
  pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
  pl.xlabel('$x$')
  pl.ylabel('$f(x)$')
  pl.ylim(-10, 20)
  pl.legend(loc='upper left')
  pl.show()

  l = gaussianP.loglikelihood(np.array([1.0, 1.0]))
  print l

  lg = gaussianP.loglikilhoodgrad(np.array([1.0, 1.0]))
  print lg

  print gaussianP.optimizehyperparams()


def main():
  gpTest()

if __name__ == '__main__':
  main()