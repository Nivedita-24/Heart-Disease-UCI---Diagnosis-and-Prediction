## PROJECT SUMMARY AND TASK DETAIL

library(dummies)
library(caret)
library(FNN)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(adabag)
library(plotly)
library(ggpubr)
library(gains)
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
summary(Heart) # The dataset is balanced

#----------------------------------------------------------------------------------------------
# Data Cleaning and Pre-processing

Heart <- na.omit(Heart) # Removing missing values (NA)
dim(Heart)
summary(Heart)

Heart$HasDisease <- 1*(Heart$AHD =="Yes") # Creating a new variable HasDisease.

Heart <- Heart[,-c(1,15)] # Removing Index column and AHD Column
summary(Heart)

summary(Heart)
#----------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION MODEL

# Changing categorical variables to factors
HeartLR <- Heart
HeartLR[,c("Sex","Fbs","ExAng","RestECG","Slope","Ca")] <- lapply(HeartLR[,c("Sex","Fbs","ExAng","RestECG","Slope","Ca")],factor)
summary(HeartLR)

HeartLR[,"Sex"] <- factor(HeartLR$Sex, levels=c(1,0), labels =c("Male","Female"))

## Data Visualization
## Continuous Variables

plot1 <- ggplot(HeartLR, aes(x=as.factor(HeartLR$HasDisease), y=Age)) + 
  geom_boxplot(fill="lightcoral") + labs(x="Has Disease", y = "Age (Years)")

plot2 <- ggplot(HeartLR, aes(x=as.factor(HeartLR$HasDisease), y=RestBP)) + 
  geom_boxplot(fill="lightcoral") + labs(x="Has Disease", y = "Resting blood pressure in mm")

plot3 <- ggplot(HeartLR, aes(x=as.factor(HeartLR$HasDisease), y=Chol)) + 
  geom_boxplot(fill="lightcoral") + labs(x="Has Disease", y = "Serum cholestoral in mg/dl")

plot4 <- ggplot(HeartLR, aes(x=as.factor(HeartLR$HasDisease), y=MaxHR)) + 
  geom_boxplot(fill="lightcoral") + labs(x="Has Disease", y = "maximum heart rate achieved")

plot5 <- ggplot(HeartLR, aes(x=as.factor(HeartLR$HasDisease), y=Oldpeak)) + 
  geom_boxplot(fill="lightcoral") + labs(x="Has Disease", y = "ST depression induced by exercise relative to rest")

figure <- ggarrange(plot1, plot2, plot3, plot4, plot5,
                    labels = c("Age", "RestBP", "Chol","MaxHR","Oldpeak"),
                    ncol = 5, nrow = 1)
figure

## OBSERVATION : 
# Most people who are suffering from heart disease are aged, having high cholestrol, high ST depression and low heart rate.

## Categorical Variables

data.for.plot1 <- aggregate(HeartLR$HasDisease, by = list(HeartLR$Sex), FUN = mean)
names(data.for.plot1) <- c("Sex", "AvgHeartPatients")
plot6 <- ggplot(data.for.plot1) + geom_bar(aes(x = Sex, y = AvgHeartPatients), stat = "identity", color ="cyan4", fill = "lightcyan2")

data.for.plot2 <- aggregate(HeartLR$HasDisease, by = list(HeartLR$ChestPain), FUN = mean)
names(data.for.plot2) <- c("ChestPain", "AvgHeartPatients")
plot7 <- ggplot(data.for.plot2) + geom_bar(aes(x = ChestPain, y = AvgHeartPatients), stat = "identity", color ="cyan4", fill = "lightcyan2")

data.for.plot3 <- aggregate(HeartLR$HasDisease, by = list(HeartLR$Fbs), FUN = mean)
names(data.for.plot3) <- c("Fbs", "AvgHeartPatients")
plot8 <- ggplot(data.for.plot3) + geom_bar(aes(x = Fbs, y = AvgHeartPatients), stat = "identity", color ="cyan4", fill = "lightcyan2")

data.for.plot4 <- aggregate(HeartLR$HasDisease, by = list(HeartLR$RestECG), FUN = mean)
names(data.for.plot4) <- c("RestECG", "AvgHeartPatients")
plot9 <- ggplot(data.for.plot4) + geom_bar(aes(x = RestECG, y = AvgHeartPatients), stat = "identity", color ="cyan4", fill = "lightcyan2")

data.for.plot5 <- aggregate(HeartLR$HasDisease, by = list(HeartLR$Thal), FUN = mean)
names(data.for.plot5) <- c("Thal", "AvgHeartPatients")
plot10 <- ggplot(data.for.plot5) + geom_bar(aes(x = Thal, y = AvgHeartPatients), stat = "identity", color ="cyan4", fill = "lightcyan2")

figure2 <- ggarrange(plot6, plot7, plot8, plot9,plot10,
                    labels = c("Sex", "ChestPain", "Fbs","RestECG","Thal"),
                    ncol = 5, nrow = 1)
         

## OBSERVATION : 
# Most people who are suffering from heart disease are males, having high resting ECG, high fasting blood sugar, asymptomatic chest pain and reversible defect.

#----------------------------------------------------------------------------------------------
# Partitioning the data.

# randomly sample 50% of the row IDs for training, 30% serve as validation and the remaining 20% as testing data.
set.seed(1)

train.index<-sample(rownames(HeartLR),dim(HeartLR)[1]*0.5)
valid.index<-sample(setdiff(rownames(HeartLR),train.index), dim(HeartLR)[1]*0.3)
test.index<-sample(setdiff(rownames(HeartLR),union(train.index,valid.index)))

train.df<-HeartLR[train.index,]
valid.df<-HeartLR[valid.index,]
test.df<-HeartLR[test.index,]
dim(train.df)
dim(valid.df)
dim(test.df)

# running logistic regression
# use glm() (general linear model) with family = "binomial" to fit a logistic regression.
heartlogit.reg <- glm(HasDisease ~ ., data = train.df, family = "binomial") 
options(scipen=999)
summary(heartlogit.reg)

Heart.reg.pred <- predict(heartlogit.reg, valid.df[,-14], type = "response") 
confusionMatrix(as.factor(ifelse(Heart.reg.pred > 0.5, 1, 0)), as.factor(valid.df$HasDisease))

# Accuracy in prediction is 85.39% using Logistic Regression Model.
#----------------------------------------------------------------------------------------------
# KNN CLASSIFICATION MODEL

# Converting categorical variables to dummy variables
Heart_df <- dummy.data.frame(Heart) 

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
# DECISION TREE MODEL

# Partitioning the data.
# randomly sample 50% of the row IDs for training, 30% serve as validation and the remaining 20% as testing data.
set.seed(1)

train.index<-sample(rownames(Heart),dim(Heart)[1]*0.5)
valid.index<-sample(setdiff(rownames(Heart),train.index), dim(Heart)[1]*0.3)
test.index<-sample(setdiff(rownames(Heart),union(train.index,valid.index)))

train.df<-Heart[train.index,]
valid.df<-Heart[valid.index,]
test.df<-Heart[test.index,]
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
# RANDOM FOREST MODEL

Heartrf <- randomForest(as.factor(HasDisease) ~ ., data = train.df, ntree = 500, 
                       mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(Heartrf, type = 1)

# OBSERVATION
# The most important variables, in identifying whether a patient is suffering from heart disease, are Ca, Sex, Thal, Oldpeak, ExAng, Max HR and Slope.

## confusion matrix of the random forest above, use the valid data

Heartrf.pred.valid <- predict(Heartrf,valid.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(Heartrf.pred.valid, as.factor(valid.df$HasDisease))

# Accuracy in prediction is 79.8% using Random forest trees Model.

#----------------------------------------------------------------------------------------------
# BOOSTED TREES MODEL

set.seed(1)

train.df$HasDisease <- as.factor(train.df$HasDisease)
heartboost <- boosting(HasDisease ~ ., data = train.df)

#Predict using Valid data

heartBoost.pred.valid <- predict(heartboost,valid.df,type = "class")
# generate confusion matrix for training data
confusionMatrix(as.factor(heartBoost.pred.valid$class), as.factor(valid.df$HasDisease))

# Accuracy in prediction is 78.6 % using Random forest trees Model.
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

## FINAL RECOMMENDED MODEL

# Final Recommended model is logistic regression model with an accuracy of 85.39%.
# It is interesting to note that KNN also has 85% accuracy, we would however not go with this model in this case.
# Mainly because when the model will be deployed, the dataset would be huge, So the KNN model will be slow in predicting compared to other models as it is a "lazy learner".

## Calculating Lift of Final Model.
heartgain <- gains(valid.df$HasDisease , Heart.reg.pred, groups=10)
heartgain

# plot lift chart
plot(c(0,heartgain$cume.pct.of.total*sum(valid.df$HasDisease))~c(0,heartgain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="Heart Disease UCI", type="l")
lines(c(0,sum(valid.df$HasDisease))~c(0, dim(valid.df)[1]), lty=2)

# The “lift” over the base curve indicates for a given number of cases (read on the x-axis), the additional patients that we can identify by using the model.

# compute deciles and plot decile-wise chart
heights <- heartgain$mean.resp/mean(valid.df$HasDisease)
heights
decileplot <- barplot(heights, names.arg = heartgain$depth, ylim = c(0,4), 
                      xlab = "Percentile", ylab = "Mean Response/Overall Mean", main = "Decile-wise lift chart")

# add labels to columns
text(decileplot, heights+0.5, labels=round(heights, 1), cex = 0.8)

# Taking 10% of the records that are ranked by the model as “most probable 1’s” yields 2 times as many 1’s as would simply selecting 10% of the records at random. 


