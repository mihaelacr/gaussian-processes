"""The main file that tests """

import numpy as np
import gp

# Let's do some plotting
from matplotlib import pyplot as pl

def gpTest():

  def f(x):
      """The function to predict."""
      return x * np.sin(x)
  X = np.array([1., 3., 5., 6., 7., 8.])

  print X
  # Observations
  y = f(X)
  print y
  meanY = np.mean(y)

  # you have to subtract the mean to kind of make sure they have mean 0
  y = y - meanY

  gaussianP = gp.GaussianProcess(covFunction=gp.CubicExponential())
  gaussianP.fit(X, y)
  res =  gaussianP.predict(0.0)
  print "predict"
  print res

  # Plot the function, the prediction and the 95% confidence interval based on
  # the MSE
  x = np.atleast_2d(np.linspace(0, 10, 1000)).T
  # my predict still does not work with mutiple instances but doing that is not hard
  y_pred = gaussianP.predictAll(x)

  y_pred = y_pred + meanY
  print y_pred

  fig = pl.figure()
  pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
  pl.plot(X, f(X), 'r.', markersize=10, label=u'Observations')
  pl.plot(x, y_pred, 'b-', label=u'Prediction')
  pl.xlabel('$x$')
  pl.ylabel('$f(x)$')
  pl.ylim(-10, 20)
  pl.legend(loc='upper left')
  pl.show()

def main():
  gpTest()

if __name__ == '__main__':
  main()