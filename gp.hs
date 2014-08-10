-- I was thinking of creating a GP data type but it seems that the HasGP guy does not have this

import Data.Vector (Vector)
import qualified Numeric.Container as NC
import qualified Data.Vector as V
import qualified Data.Packed.Matrix as M
import qualified Numeric.LinearAlgebra.HMatrix as LA

type CovarianceFunction = (Double, Double) -> Double


--k_data_data :: M.Matrix Double = cmap cov $ NC.outer x_data x_data

predict :: CovarianceFunction -> Vector (Double, Double) -> Double -> (Double, Double)
predict cov input to_predict =
  let (x_data, y_data) = V.unzip input
      k_data_data  = M.mapMatrix cov $ M.fromLists [[(x1, x2) | x1 <-V.toList x_data] | x2 <- V.toList x_data]
      k_data_predict = NC.cmap cov $ V.fromList [(x, to_predict) | x <- x_data]
      k_predict_data = NC.cmap cov $ V.fromList [(to_predict, x) | x <- x_data]
      k_predict_predict = cov (to_predict, to_predict)
      inv_k_data_data = LA.inv k_data_data
      prediction_mean = foldl LA.dot [k_predict_data, inv_k_data_data, y_data]
      prediction_covariance = k_predict_predict - foldl LA.dot [k_predict_data, inv_k_data_data,  k_data_predict]
  in (prediction_mean, prediction_covariance)


squareExponential :: CovarianceFunction
squareExponential (x1, x2) = exp (- (x1 - x2) ** 2)

cubicExponential :: CovarianceFunction
cubicExponential (x1, x2) = exp (- abs (x1 - x2) ** 3)

main :: IO ()
main = do
  let f x = x * sin x
      gpData = V.fromList [(x, f x) | x <- [1, 3, 5, 6, 7, 8] ]
      prediction = predict squareExponential gpData 0
  print prediction
