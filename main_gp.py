"""The main file that tests """

import numpy as np
import gp

def gpTest():

  def f(x):
      """The function to predict."""
      return x * np.sin(x)
  x = np.array([1., 3., 5., 6., 7., 8.])

  print x
  # Observations
  y = f(x)
  print y
  meanY = np.mean(y)

  # you have to subtract the mean to kind of make sure they have mean 0
  y = y - meanY

  gaussianP = gp.GaussianProcess(covFunction=gp.SquaredExponential())
  gaussianP.fit(x, y)
  res =  gaussianP.predict(0.0)
  print "predict"
  print res

def main():
  gpTest()

if __name__ == '__main__':
  main()