pop_data <- read.csv("/home/shreyashkharat/Datasets/diabetes.csv", header = TRUE)
pop_data_fixed <- read.csv("/home/shreyashkharat/Datasets/diabetes.csv", header = TRUE)

# Train Test split
require("caTools")
set.seed(0)
split = sample.split(pop_data, SplitRatio = 0.8)
train_set = subset(pop_data, split == TRUE)
test_set = subset(pop_data, split == FALSE)

# Classification Tree Building
require("rpart")
require("rpart.plot")
# Tree buliding
class_tree <- rpart(formula = Outcome~., data = train_set, method = "class", control = rpart.control(maxdepth = 5))
# Plot of tree
rpart.plot(class_tree, box.palette = "RdYlGn", digits = -3)
# Predict values
class_predict <- predict(class_tree, test_set, type = "class")
table(class_predict, test_set$Outcome)
(94+33)/170
# The Simple Classification Tree gives an accuracy of 0.7470

# Pruning class_tree and full tree Building
full_tree = rpart(formula = Outcome~., data = train_set, method = "class", control = rpart.control(cp = 0))
# Plot of full_tree
rpart.plot(full_tree, box.palette = "RdYlGn", digits = -3)
# Finding optimum cp
min_cp <- class_tree$cptable[which.min(class_tree$cptable[, "xerror"]), "CP"]
# Building pruned tree
pruned_tree <- prune(full_tree, min_cp)
# Assess accuracy of pruned_tree and full_tree.
pruned_predict <- predict(pruned_tree, test_set, type = "class")
full_predict <- predict(full_tree, test_set, type = "class")
# Confusion Matrix 
table(pruned_predict, test_set$Outcome)
table(full_predict, test_set$Outcome)
(91+35)/170
# Pruned tree gives an accuracy of 0.7411.
(87+35)/170
# Full Tree gives an accuracy of 0.7176

# Let's try Ensemble Techniques
# Ensemble Technique: 1. BAGGING
require("randomForest")
set.seed(0)
bagging_model = randomForest(formula = Outcome~., data = train_set, mtry = 8)
bagging_probs <- predict(bagging_model, test_set)
bagging_predict <- rep("NO", 170)
bagging_predict[bagging_probs > 0.5] <- "YES"
# Confusion Matrix
table(bagging_predict, test_set$Outcome)
(90+35)/170
# The bagging model gives an accuracy of 0.7352.

# Ensemble Technique: 2. RANDOM FOREST
set.seed(0)
random_forest_model = randomForest(formula = Outcome~., data = train_set, ntree = 500, mtry = 3)
random_forest_probs <- predict(random_forest_model, test_set)
random_forest_predict <- rep("NO", 170)
random_forest_predict[random_forest_probs > 0.5] <-"YES"
# Confusion Matrix
table(random_forest_predict, test_set$Outcome)
(92+35)/170
# The Random Forest Model gives an accuracy of 0.7470, which nearly that of Simple tree.

# Ensemble Technique: 3. GBM or Gradient Boosting Model
require("gbm")
set.seed(0)
# The data is already gbm compatible.
# Model building
model_gbm <- gbm(Outcome~., data = train_set, distribution = "bernoulli", n.trees = 5000, interaction.depth = 18, shrinkage = 0.1, verbose = FALSE)
# For Regression models use Gaussian distribution and for classification models use Bernoulli distribution.
gbm_probs <- predict(model_gbm, test_set, type = "response")
gbm_predict <- rep("NO", 170)
gbm_predict[gbm_probs > 0.5] <- "YES"
# Confusion Matrix
table(gbm_predict, test_set$Outcome)
(92+35)/170
# The Random Forest Model gives an accuracy of 0.7470, which nearly that of Simple tree.

# Ensemble Technique: 4. AdaBoost or Adaptive Boost
require("adabag")
train_set$Outcome <- as.factor(train_set$Outcome)
set.seed(0)
model_adaboost <- boosting(Outcome~., data = train_set, boos = TRUE, mfinal = 1000)
ada_predict <- predict(model_adaboost, test_set)
table(ada_predict$class, test_set$Outcome)
(87+35)/170
# AdaBoost gives an accuracy of 0.7176.

# Ensemble Technique: 5. XGBoost or Xtreme Gradient Boosting Model
require("xgboost")
# For XGBoost in classification models we need boolean dependent variable
train_Y <- train_set$Outcome == "1"
test_Y <- test_set$Outcome == "1"
# Also, the dependent variables need to be in model matrix form ie free form categorical variables
train_X <- model.matrix(Outcome~.-1, data = train_set)
train_X <- train_X[, -12]
test_X <- model.matrix(Outcome~.-1, data = test_set)
test_X <- test_X[, -12]
# Creating DMatrices for XGBoost Model
DMatrix_train <- xgb.DMatrix(data = train_X, label = train_Y)
DMatrix_test <- xgb.DMatrix(data = test_X, label = test_Y)
# Model building
model_xgboost <- xgboost(data = DMatrix_train, nrounds = 50, objective = "multi:softmax", eta = 0.3, num_class = 2)
# Prediction
xgboost_predict <- predict(model_xgboost, DMatrix_test)
# Confusion Matrix
table(xgboost_predict, test_Y)
(90+34)/170
# XGBoost gives an accuracy of 0.7294.
# The highest accuracy is given by random forest and simple deecision tree as 0.7470.