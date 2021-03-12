library("caret")

zScoreNormalize <- function(data){
  iris_numeric <- iris[1:4]
  data = iris_numeric
  
  pp_unit <- preProcess(data, method = c("range"))
  iris_numeric_unit <- predict(pp_unit, data)
  pp_zscore <- preProcess(data, method = c("center", "scale"))
  iris_numeric_zscore <- predict(pp_zscore, data)
  print(iris_numeric_zscore) 
  
}


zScoreNormalize(x)