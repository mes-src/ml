library(caret)
data(cars)



#head(cars)
#____________PreProcessing____________
cars_cor <- cor(cars)
print(cars_cor)
findCorrelation(cars_cor, cutoff=0.75)
findLinearCombos(cars)

cars <- cars[,c(-15,18)]
set.seed(232455)
cars_sampling_vector <- createDataPartition(cars$Price, p=0.85, list = FALSE)
cars_train <- cars[cars_sampling_vector,]
cars_train_features <- cars[,-1]
cars_train_labels <- cars$Price[cars_sampling_vector]
cars_test <- cars[-cars_sampling_vector,]
cars_test_labels <- cars$Price[-cars_sampling_vector]



#____________TRAINING____________

cars_model <- lm(Price ~ . -Saturn, data = cars_train) 
#How does the price of the car vary based uponn all the rest of the features in the dataset [i.e. cylinders, doors, sedan,brand, etc]
alias(cars_model) #Exclude Saturn as a feature due to aliasing
summary(cars_model)
summary(cars_model$residuals) #residual = actual value of obs - predicted value of observation
mean(cars_train$Price)


par(mfrow = c(2,1))#Plot
cars_residuals <- cars_model$residuals
qqnorm(cars_residuals, main = "Normal QQ plot for Cars dataset")
qqline(cars_residuals)


n_cars <- nrow(cars_train)
k_cars <- length(cars_model$coefficients) -1 
RSE <- sqrt(sum(cars_model$residuals ^2) / (n_cars - k_cars - 1))
print(RSE)
print(mean(cars_train$Price)) #how far away the regression line will be on average


compute_Rsq <- function(x,y){
  rss <- sum((x-y)^2)
  tss <- sum((y - mean(y)) ^2)
  return(1 - (rss /tss))
}
r2 <- compute_Rsq(cars_model$fitted.values, cars_train$Price)
print(r2) #Square of the correlation bw oputput variable and input feature



#____________TESTING_____________

cars_model_predictions <- predict(cars_model, cars_test)

computeMSE <- function(prediction, actual){
  mean( ( prediction - actual) ^2)
}
computeMSE(cars_model$fitted.values, cars_train$Price)
computeMSE(cars_model_predictions, cars_test$Price)





