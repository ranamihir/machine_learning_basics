install.packages("class")
library("class")

iris.train <- read.csv("iris_train.csv", header = TRUE)
iris.train$ID = NULL

iris.test <- read.csv("iris_test.csv", header = TRUE)
final = as.data.frame(iris.test$ID)
iris.test$ID = NULL

iris.trainLabels <- iris.train[, 5]

iris_pred <- knn(train = iris.train[, 1:4], test = iris.test, cl = iris.trainLabels, k=3)

final$Species = as.matrix(iris_pred)

names(final) <- c("ID", "Species")

write.csv(final, file = 'solution.csv', row.names = FALSE)