library("caret")


#Remove highly Correlated Features
iris_cor <- cor(iris_numeric)
findCorrelation(iris_cor, cutoff = 0.8) #specify correlation cutoff threashold - returns column index of columns to remove
print(findCorrelation)