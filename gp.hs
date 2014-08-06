-- I was thinking of creating a GP data type but it seems that the HasGP guy does not have this

import Data.Vector (Vector)
import qualified Data.Vector as V

type CovarianceFunction = (Double, Double) -> Double

predict :: CovarianceFunction -> Vector (Double, Double) -> (Double, Double)
predict = undefined



squareExponential :: CovarianceFunction
squareExponential (x1, x2) = exp (- (x1 - x2) ** 2)

cubicExponential :: CovarianceFunction
cubicExponential (x1, x2) = exp (- abs (x1 - x2) ** 3)

main :: IO ()
main = do
  let f x = x * sin x
      gpData = V.fromList [(x, f x) | x <- [1, 3, 5, 6, 7, 8] ]
      prediction = predict squareExponential gpData
  print prediction
