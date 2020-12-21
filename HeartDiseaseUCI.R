## PROJECT SUMMARY AND TASK DETAIL

library(dummies)
library(caret)
library(FNN)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(adabag)
#----------------------------------------------------------------------------------------------
# Predict based on the given attributes of a patient that whether that particular person has a heart disease or not.
# Diagnose and find out various insights from this dataset which could help in understanding the problem more

#----------------------------------------------------------------------------------------------
# Reading the file
Heart <- read.csv("Heart.csv")
dim(Heart) # 303 Patient's record.
str(Heart)
head(Heart)

# Dataset Variables
# Index
# Age: Age in years
# Sex: Sex (1 = male, 2 = female)
# ChestPain: Chest Pain Type
# RestBP: Resting blood pressure (in mm Hg on admission to the hospital)
# Chol: Serum cholestoral in mg/dl
# Fbs: Fasting blood sugar &gt; 120 mg/dl (1 = true; 0 = false)
# RestECG: Resting electrocardiographic results
# MaxHR: maximum heart rate achieved
# ExAng: exercise induced angina (1 = yes; 0 = no)
# Oldpeak: ST depression induced by exercise relative to rest
# Slope: The slope of the peak exercise ST segment
# Ca: Number of major vessels (0-3) colored by flourosopy
# Thal: normal, fixed defect, reversable defect
# AHD: Yes, No
#----------------------------------------------------------------------------------------------
# Summarising the dataset
summary(Heart)

# Data Cleaning and Pre-processing

HeartCD <- na.omit(Heart) # Removing missing values (NA)
dim(HeartCD)
summary(HeartCD)

HeartCD$HasDisease <- 1*(HeartCD$AHD =="Yes") # Creating a new variable HasDisease.

HeartCD <- HeartCD[,-c(1,15)] # Removing Index column and AHD Column
summary(HeartCD)

Heart_df <- dummy.data.frame(HeartCD) # Converting categorical variables to dummy variables
summary(Heart_df)

#----------------------------------------------------------------------------------------------

# Partitioning the data.
# randomly sample 50% of the row IDs for training, 30% serve as validation and the remaining 20% as testing data.
set.seed(1)

train.index<-sample(rownames(Heart_df),dim(Heart_df)[1]*0.5)
valid.index<-sample(setdiff(rownames(Heart_df),train.index), dim(Heart_df)[1]*0.3)
test.index<-sample(setdiff(rownames(Heart_df),union(train.index,valid.index)))

train.df<-Heart_df[train.index,]
valid.df<-Heart_df[valid.index,]
test.df<-Heart_df[test.index,]
dim(train.df)
dim(valid.df)
dim(test.df)

#----------------------------------------------------------------------------------------------
# KNN Classification Model

# Normalizing the data before running KNN Model
train.norm.df <- train.df
valid.norm.df <- valid.df
test.norm.df <- test.df

dim(train.norm.df)
norm.values <- preProcess(train.df[,c(1:18)],method=c("center", "scale"))

train.norm.df[,c(1:18)] <- predict(norm.values,train.df[,c(1:18)])
valid.norm.df[,c(1:18)] <- predict(norm.values,valid.df[,c(1:18)])
test.norm.df[,c(1:18)] <- predict(norm.values,test.df[,c(1:18)])

summary(train.norm.df)

# Choosing the value for 'k'
accuracy.df <- data.frame( k = seq(1,20,1), accuracy = rep(0,20))
accuracy.df

for (i in 1:20){
  knn.pred <-knn(train.norm.df[,1:18],valid.norm.df[,1:18], cl=train.norm.df[,19]
                 , k=i)
  accuracy.df[i,2]<- confusionMatrix(factor(knn.pred), factor(valid.norm.df[,19]))$overall[1]
}

accuracy.df # accuracy is best at k=17

# Checking on test dataset

knn.pred.test <-knn(train.norm.df[,1:18], test.norm.df[,1:18], cl=train.norm.df[,19]
                    ,k=17)

confusionMatrix(factor(knn.pred.test),factor(test.norm.df[,19]))

# Accuracy in prediction is 85% using KNN Classification Model.

#----------------------------------------------------------------------------------------------
# Classification Trees Model

# Partitioning the data.
# randomly sample 50% of the row IDs for training, 30% serve as validation and the remaining 20% as testing data.
set.seed(1)

train.index<-sample(rownames(HeartCD),dim(HeartCD)[1]*0.5)
valid.index<-sample(setdiff(rownames(HeartCD),train.index), dim(HeartCD)[1]*0.3)
test.index<-sample(setdiff(rownames(HeartCD),union(train.index,valid.index)))

train.df<-HeartCD[train.index,]
valid.df<-HeartCD[valid.index,]
test.df<-HeartCD[test.index,]
dim(train.df)
dim(valid.df)
dim(test.df)

# Running a full grown classification tree
heart_tree <- rpart(HasDisease ~., data=train.df, method='class', cp=0, minsplit=1, xval=5)

# using prp to draw this tree
prp(heart_tree, type = 1, extra = 1, split.font = 1, varlen = -10)

# prune by lower cp
heartpruned.ct <- prune(heart_tree, 
                       cp = heart_tree$cptable[which.min(heart_tree$cptable[,"xerror"]),"CP"])

#Plot the pruned tree
prp(heartpruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(heartpruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

#To find a better parsimonous pruned tree

heart_tree$cptable
# xerror + xstd
# This is the best pruned tree.

heart.ct.predict <- predict(heartpruned.ct, valid.df, type='class')
confusionMatrix(heart.ct.predict, as.factor(valid.df$HasDisease))

# Accuracy in prediction is 72% using Classification trees Model.

#----------------------------------------------------------------------------------------------
# Random Forest Model

Heartrf <- randomForest(as.factor(HasDisease) ~ ., data = train.df, ntree = 500, 
                       mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(Heartrf, type = 1)

## confusion matrix of the random forest above, use the valid data

Heartrf.pred.valid <- predict(Heartrf,valid.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(Heartrf.pred.valid, as.factor(valid.df$HasDisease))

# Accuracy in prediction is 79.8% using Random forest trees Model.

#----------------------------------------------------------------------------------------------
# Boosted Trees

set.seed(1)

train.df$HasDisease <- as.factor(train.df$HasDisease)
heartboost <- boosting(HasDisease ~ ., data = train.df)

#Predict using Valid data

heartBoost.pred.valid <- predict(heartboost,valid.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(as.factor(heartBoost.pred.valid$class), as.factor(valid.df$HasDisease))

# Accuracy in prediction is 78.6 % using Random forest trees Model.
