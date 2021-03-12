library("caret")
library("e1071")

iris_sampling_vector <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
iris_test <- iris_numeric[-iris_sampling_vector,]
iris_test_z <- iris_numeric_zscore[-iris_sampling_vector,]
iris_test_labels <- iris$Species[-iris_sampling_vector]

#Knn
knn_model <- knn3(iris_train, iris_train_labels, k=5)
knn_predictions_prob <- predict(knn_model,iris_test, type="prob")
tail(knn_predictions_prob)

knn_predictions <-predict(knn_model, iris_test, type ="class")

postResample(knn_predictions, iris_test_labels) #Prediction Accuracy
table(knn_predictions, iris_test_labels) #total number of correct classifications, and mistakes

