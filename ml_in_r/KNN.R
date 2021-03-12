head(iris)
#Calculate Euclidean Distance

iris_features <- iris[1:4]
eucl_dist <- function(x1, x2) sqrt (sum((x1-x2) ^ 2))
distances <- apply(iris_features, 1, 
                   function(x) eucl_dist(x, iris_features)) 
distances_sorted <- sort(distances, index.return =T) 
#Apply Euclidean Distance Formula of a given Point x, to every other point
#Sort the distances so the nearest neighbors are at the top
print(distances_sorted)

#$x value represents the actual value of the distacnes computed bw our sample iris flower and the observations in the data
#$ix cotntains the row numbers of the corresponding observation
knn = iris[distances_sorted$ix[1:5],]
#Subsets the dataftrame to return the K neirest neighbors
print(knn)




