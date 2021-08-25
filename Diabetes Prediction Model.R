# Required packages: ggplot2, MASS, caTools, class.
pop_data <- read.csv("/home/shreyashkharat/Datasets/diabetes.csv", header = TRUE)
pop_data_fixed <- read.csv("/home/shreyashkharat/Datasets/diabetes.csv", header = TRUE)
summary(pop_data)
hist(pop_data$Insulin)
hist(pop_data$DiabetesPedigreeFunction)
hist(pop_data$BMI)

# From the summary, it's clear that Insulin and DiabetesPedigreeFunction has outliers.

# Outlier treatment for variable Insulin.
max_Insulin <- quantile(pop_data$Insulin, 0.9)
pop_data$Insulin[pop_data$Insulin > max_Insulin] <- max_Insulin
summary(pop_data$Insulin)

# Outlier treatment for variable DiabetesPedigreeFunction.
max_DPG <- quantile(pop_data$DiabetesPedigreeFunction, 0.99)
pop_data$DiabetesPedigreeFunction[pop_data$DiabetesPedigreeFunction > max_DPG] <- max_DPG
summary(pop_data$DiabetesPedigreeFunction)

# Logistic Regression Model
model_logi <- glm(Outcome~., data = pop_data, family = binomial)
summary(model_logi)
# Probabilities
logi_probs <- predict(model_logi, type = "response")
# Prediction array
logi_predict <- rep("NO", 768)
logi_predict[logi_probs > 0.5] <- "YES"
# Confusion Matrix
table(logi_predict, pop_data$Outcome)
(445+156)/768
# The logistic Regression model gives an R^2 of 0.7825.

# Linear Discriminant Analysis
require("MASS")
model_lda <- lda(Outcome~., data = pop_data)
# Probabilities
lda_probs <- predict(model_lda, pop_data)
# Prediction array
lda_predict <- lda_probs$class
# Confusion Matrix
table(lda_predict, pop_data$Outcome)
(444+155)/768
# The Linear Discriminant Analysis model gives an R^2 of 0.7799.

# Quadratic Discriminant Analysis
model_qda <- qda(Outcome~., data = pop_data)
# Probabilities
qda_probs <- predict(model_qda, pop_data)
# Prediction array
qda_predict <- qda_probs$class
# Confusion Matrix
table(qda_predict, pop_data$Outcome)
(427+162)/768
# The Quadratic Discriminant Analysis model gives an R^2 of 0.7669.

# Lets check the accuracy of the above models on test sets.

# Train Test Split
require("caTools")
set.seed(0)
pop_split = sample.split(pop_data, SplitRatio = 0.8)
training_set <- subset(pop_data, pop_split ==TRUE)
test_set <- subset(pop_data, pop_split == FALSE)

# Logistic Regression Model
model_logi_train <- glm(Outcome~., data = training_set, family = binomial)
# Probabilities
logi_train_probs <- predict(model_logi_train, test_set, type = "response")
# Prediction array
logi_train_predict <- rep("NO", 170)
logi_train_predict[logi_train_probs > 0.5] <- "YES"
# Confusion Matrix
table(logi_train_predict, test_set$Outcome)
(98+32)/170
# On this test set, Logistic Regression Model gives an R^2 of 0.7647.

# Linear Discriminant Analysis
model_lda_train <- lda(Outcome~., data = training_set)
# Probabilities
lda_train_probs <- predict(model_lda_train, test_set)
# Prediction array
lda_train_predict <- lda_train_probs$class
# Confusion Matrix
table(lda_train_predict, test_set$Outcome)
(99+32)/170
# On this test set, Linear Discriminant Analysis model gives an R^2 of 0.7705.

# Quadratic Discriminant Analysis
model_qda_train <- qda(Outcome~., data = training_set)
# Probabilities
qda_train_probs <- predict(model_qda_train, test_set)
# Prediction array
qda_train_predict <- qda_train_probs$class
# Confusion Matrix
table(qda_train_predict, test_set$Outcome)
(95+34)/170
# On this test set, Quadratic Discriminant Analysis model gives an R^2 of 0.7588.

# K Nearest Neighbor 
require("class")
# Function arguments
train_x <- training_set[, -9]
train_y <- training_set$Outcome
test_x <- test_set[, -9]
test_y <- test_set$Outcome
# Scaling of train_x, test_x.
train_x_scale <- scale(train_x)
test_x_scale <- scale(test_x)
# Function declaration
knn_model <- knn(train_x_scale, test_x_scale, train_y, k = 30)
table(knn_model, test_set$Outcome)
(102+30)/170
# On this test set, K Nearest Neighbor Model gives an R^2 of 0.7764.
# We get the highest accuracy in K Nearest Neighbor model for the above test set.
