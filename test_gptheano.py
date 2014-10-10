from gptheano import *
import theano
import numpy as np

def covarianceMatrix(xs, func):
    return np.array([func(x1, x2) for x1 in xs for x2 in xs]).reshape((len(xs), len(xs)))

def squaredExp(x1, x2):
  return np.exp(-(x1- x2)**2)

def testNumpyEquivalence():
  exp = SquaredExponential()

  xs = T.dmatrix('xs')
  mat = exp.covarianceMatrix(xs)

  fun = theano.function(inputs=[xs], outputs=mat, updates={})

  inputs = [np.array([[1], [2]]), np.array([[-3.], [0.], [2.]]), np.array([[1.0/2], [1.0/3], [-1.0/12], [-1.5]])]

  for var in inputs:
    print "testing", var
    res = fun(var)
    assert np.allclose(res, covarianceMatrix(var, squaredExp))

def main():
  testNumpyEquivalence()

if __name__ == '__main__':
  main()
