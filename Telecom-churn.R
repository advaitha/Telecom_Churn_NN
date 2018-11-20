#Data understanding, preparation and cleaning
# To install the package if not already installed, 
# if installed then will only load the library of that package
if (!"install.load" %in% rownames(installed.packages()))
  install.packages("install.load")
library(install.load)
#install the required packages
pkgs_to_install_load <- c("readr","dplyr","tidyr","lubridate","ggplot2","car","h2o",
                          "caTools","MASS", "gridExtra", "e1071", "klaR", "arules", "class", "scales", 
                          "purrr","kernlab","doParallel","png","caret","h2o")
sapply(pkgs_to_install_load,install_load)


#As this assignment is computationally intensive,use all the cores of cpu
registerDoParallel(cores = 8)

#load the dataset
churn <- read.csv("telecom_nn_train.csv",stringsAsFactors = FALSE)
#Find NA in the dataset
sum(is.na(churn))
#06 NA found in the dataset

# Let's find out the column in which we are getting missing value
sapply(churn, function(x) sum(is.na(x)))

# Missing values are present in Total charge only
# After analysing the data it is found that these  rows contains data for those customers
# Who are not getting churned
# So removing those rows as these rows are not much significant for the analysis
# and a very less numbers as compare to all records
churn <- churn %>% na.omit()

# Let's us run the is.na function again to check if missing values got ommited or not
sum(is.na(churn)) # 0
# No missing values

#Check for depulicated observations
nrow(unique(churn))==nrow(churn)
sum(duplicated(churn)) #10
# No of unique rows and total records are not equal 
# 10  duplicate observations are present

#Remove the duplicated observations
churn <- subset(churn,(!duplicated(churn)==TRUE))
#Again check for duplicate observations
sum(duplicated(churn)) #0

#Function for creating plot. 
churn_bargraph <- function(z, na.rm = TRUE, ...) {
  nm <- names(z)
  for (i in seq_along(nm)) {
    plots <-ggplot(z,aes_string(x=nm[i],fill=factor(z$Churn))) + geom_bar(position = "fill")+
      guides(fill=guide_legend(reverse=TRUE))+
      scale_fill_discrete(labels=c("Good Customer","Churned Customer"))+
      labs(fill='churn status')
    ggsave(plots,width = 20, height = 8, units = "cm",filename=paste("myplot",nm[i],".png",sep=""))
  }
}

#Plots created will be saved in the working directory. Plots are created
#for all categorical variables. 
churn_bargraph(churn[,-c(5,8,9)])

# Analysis on numeric data
ggplot(churn, aes(x= churn$Churn, y= churn$tenure)) + geom_boxplot()
# A person who is churing, their tenure are comparatively lesser as compare to those who are not churning

ggplot(churn, aes(x= churn$Churn, y= churn$MonthlyCharges)) + geom_boxplot()
# Monthly charges for churning customers are higher as compare to non churning customers

ggplot(churn, aes(x= churn$Churn, y= churn$TotalCharges)) + geom_boxplot()
# Total charges for churning customers are lesser than compared to non churning customers

#Check the structure of the dataframe
str(churn)
glimpse(churn)
table(churn$Churn)

#For H2o package the labeled outcome needs to be factor
#convert the datatype of churn attribute from character to factor
churn$Churn <- as.factor(churn$Churn)

#Neural networks give good results
#when continous variables are scaled.
#so scaling of continous variables are undertaken
churn[ ,c(5,8,9)] <- scale(churn[ ,c(5,8,9)])

#Finally check if all the variables are in the 
#correct format required for analysis
glimpse(churn)
table(churn$Churn)

#Outlier detection and treatment
# We have three numeric variables i.e. tenure, monthly charges and Total charges
# We will be checking outliers for these three variables through box plot and
#boxplot.stats function

# Outliers for tenure
boxplot(churn$tenure)
boxplot.stats(churn$tenure)
# Can not see any outliers, so no need of outlier treatment

# Outliers for monthly charge
boxplot(churn$MonthlyCharges)
boxplot.stats(churn$MonthlyCharges)
# Can not see any outliers, so no need of outlier treatment

# Outlier for total charge
boxplot(churn$TotalCharges)
boxplot.stats(churn$TotalCharges)
# Can not see any outliers, so no need of outlier treatment

#As cross validation will be used, there is no need for validation set
#Sample split function is used to maintain the same proportion of 
#classes in both train and test datasets
set.seed(10)
s <- sample.split(churn,SplitRatio = .7)
train <- churn[s==T,]
test <- churn[s==F,]
#The split datasets are written to the working directory
#for importing as H2o objects
write.csv(train, file='churnTrain.csv', row.names=FALSE)
write.csv(test,file='churnTest.csv',row.names=FALSE)

#Check the proportion of class labels in both train and test datasets
table(train$Churn)
table(test$Churn)

# Initialize the h2o environment
library(h2o)
#use all cores and 3GB ram
h2o.init(nthreads = -1,min_mem_size = "3g") 
#check the connection
h2o.getConnection()

#Import the train and test datasets as H2o objects
churnTrain <- h2o.importFile("churnTrain.csv")
churnTest <-h2o.importFile("churnTest.csv")
class(churnTrain)

#Assign the predictor and response variables
y <- "Churn" 
x <- setdiff(names(churnTrain), y) 


#################################################################################
########## (Modeling with epoch) ##########################################
#Create an initial model. As most research papers and Andrew Ng
#suggests one hidden layer with number of neurons equal to 3 times
#number of inputs give good results, the same is tried and results checked

model_1_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(90),
                                 hidden_dropout_ratio = c(0.1),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)
model_1_epoch
model_1_epoch_predict <- h2o.predict(model_1_epoch,newdata = churnTest)
model_1_epoch_predict <- as.data.frame(model_1_epoch_predict)
head(model_1_epoch_predict)
confusionMatrix(model_1_epoch_predict$predict,test$Churn,positive = 'Yes')
#single layer with 90 neurons.
#Train - AUC 84, logloss - 5.35, Misclassification error-22.95
#CV - AUC 79.6, logloss - 1.35, Misclassification error-24.8
#Test Accuracy- 69.05%, senstivity - 83.64, Misclassification error - 30.95% 
#As the test error is high the bias of the model is high and the 
#difference between train and test is 8%, the variance is also need
#to be adjusted. First bias is reduced in the next model by increasing complexity 


model_2_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(200),
                                 hidden_dropout_ratio = c(0.1),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

model_2_epoch
model_2_epoch_predict <- h2o.predict(model_2_epoch,newdata = churnTest)
model_2_epoch_predict <- as.data.frame(model_2_epoch_predict)
head(model_2_epoch_predict)
confusionMatrix(model_2_epoch_predict$predict,test$Churn,positive='Yes')
#single layer with 200 neurons. 
#Train - AUC 83.7, logloss - 1.59, Misclassification error-24.38
#CV - AUC 78, logloss - 2.41, Misclassification error-22.8
#Test Accuracy- 66.27%, senstivity - 85.98, Misclassification error - 33.73% 
#This model decreased the accuracy with good increase in sensitivity.
#As the difference in error between test and train is more
#Regularization is applied along with increasing the layers to improve bias


model_3_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(200,200),
                                 hidden_dropout_ratio = c(0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

model_3_epoch
model_3_epoch_predict <- h2o.predict(model_3_epoch,newdata = churnTest)
model_3_epoch_predict <- as.data.frame(model_3_epoch_predict)
head(model_3_epoch_predict)
confusionMatrix(model_3_epoch_predict$predict,test$Churn,positive='Yes')
#Two layers with 200 neurons. Hidden dropout ratio is 0.2
#Train - AUC 82.1, logloss - 6.59, Misclassification error- 21%
#CV - AUC 75.5, logloss - 6.93, Misclassification error-21%
#Test Accuracy- 70.63%, senstivity - 81.54, Misclassification error - 29.4% 
#Misclassification error rate and the gap between test and train error reduced. Variance of the model also decreased.
#As there is more scope for improvement, model is further tuned
#to decrease bias and variance


model_4_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400),
                                 hidden_dropout_ratio = c(0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.1)

model_4_epoch
model_4_epoch_predict <- h2o.predict(model_4_epoch,newdata = churnTest)
model_4_epoch_predict <- as.data.frame(model_4_epoch_predict)
head(model_4_epoch_predict)
confusionMatrix(model_4_epoch_predict$predict,test$Churn,positive='Yes')
#Two layers with 200 neurons. Hidden dropout ratio is 0.2
#Train - AUC 81.4, logloss - 7.25, Misclassification error- 21.67%
#CV - AUC 72.0, logloss - 6.83, Misclassification error-20.38%
#Test Accuracy- 68.98%, senstivity - 86.45, Misclassification error - 30.5% 
#Tried to improve the model by reducing bias and variance, by increasing
#number of neurons in a layer and increasing the input dropout ratio.
#Sensitivity of the model improved at the cost of accuracy.Trying to further
#tune the model by increasing the complexity (number of layers)
#and further regularizing it (increasing input dropout ratio)


model_5_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 15,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.25)

model_5_epoch
model_5_epoch_predict <- h2o.predict(model_5_epoch,newdata = churnTest)
model_5_epoch_predict <- as.data.frame(model_5_epoch_predict)
head(model_5_epoch_predict)
confusionMatrix(model_5_epoch_predict$predict,test$Churn,positive='Yes')
#Train - AUC 77.12, logloss - 8.00, Misclassification error- 23.22%
#CV - AUC 72.2, logloss - 7.14, Misclassification error-20.68%
#Test Accuracy- 74.16%, senstivity - 81.31, Misclassification error - 25.8% 
#There is a tradeoff involved between accuracy and sensitivity.
#As the accuracy is increased sensitivity is getting reduced.
#Now keeping all the parameters same and changing the activation function
#to check the results


model_6_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="RectifierWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 15,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.25)

model_6_epoch
model_6_epoch_predict <- h2o.predict(model_6_epoch,newdata = churnTest)
model_6_epoch_predict <- as.data.frame(model_6_epoch_predict)
head(model_6_epoch_predict)
confusionMatrix(model_6_epoch_predict$predict,test$Churn,positive='Yes')
#Changing the activation function reducded the test accuracy to 65.5%
#and senstivity to 87.15%. If the objective is good accuracy model_5_epoch
#is good. 
#Further checking the impact of change in epoch on model_5_epoch


model_7_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 5,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.25)

model_7_epoch
model_7_epoch_predict <- h2o.predict(model_7_epoch,newdata = churnTest)
model_7_epoch_predict <- as.data.frame(model_7_epoch_predict)
head(model_7_epoch_predict)
confusionMatrix(model_7_epoch_predict$predict,test$Churn,positive='Yes')
#With epoch as 5 test Accuracy is 70.75% and sensitivity is 83.88%.
#These are not the best results. So changing the epoch again and checking
#the results


model_8_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 25,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.25)

model_8_epoch
model_8_epoch_predict <- h2o.predict(model_8_epoch,newdata = churnTest)
model_8_epoch_predict <- as.data.frame(model_8_epoch_predict)
head(model_8_epoch_predict)
confusionMatrix(model_8_epoch_predict$predict,test$Churn,positive='Yes')
#Accuracy of the model is 74.16 and the sensitivity is 81.31,
#Accuracy improved at the cost of sensitivity.
#Further increasing the epoch to check the results


model_9_epoch<- h2o.deeplearning(x=x,
                                 y=y,
                                 training_frame = churnTrain,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 50,
                                 keep_cross_validation_predictions = TRUE,
                                 input_dropout_ratio = 0.25)

model_9_epoch
model_9_epoch_predict <- h2o.predict(model_9_epoch,newdata = churnTest)
model_9_epoch_predict <- as.data.frame(model_9_epoch_predict)
head(model_9_epoch_predict)
confusionMatrix(model_9_epoch_predict$predict,test$Churn,positive='Yes')
#Accuracy of the model is 74.16 and the sensitivity is 81.31
#Changing the epoch from 25 to 50 doesnt make any difference.
#performace metric for test remain the same

model_10_epoch<- h2o.deeplearning(x=x,
                                  y=y,
                                  training_frame = churnTrain,
                                  seed=123,
                                  variable_importances = TRUE,
                                  activation="MaxoutWithDropout",
                                  hidden = c(400,400,400),
                                  hidden_dropout_ratio = c(0.2,0.2,0.2),
                                  l1 = 1e-5,
                                  nfolds=5,
                                  initial_weight_distribution = "Normal",
                                  balance_classes = TRUE,
                                  sparse=TRUE,
                                  reproducible = TRUE,
                                  epochs = 100,
                                  keep_cross_validation_predictions = TRUE,
                                  input_dropout_ratio = 0.25)

model_10_epoch
model_10_epoch_predict <- h2o.predict(model_10_epoch,newdata = churnTest)
model_10_epoch_predict <- as.data.frame(model_10_epoch_predict)
head(model_10_epoch_predict)
confusionMatrix(model_10_epoch_predict$predict,test$Churn,positive='Yes')
#Accuracy of the model is 74.16 and the sensitivity is 81.31
#Changing the epoch from 50 to 100 doesnt make any difference.
#performace metric for test remain the same


#As model_4_epoch gave good measure of sensitivity, regularization
#metrics were added to the model (to keep variance under check)
#and checked for any improvement
model_11_epoch <- h2o.deeplearning(x=x,
                                   y=y,
                                   training_frame = churnTrain,
                                   seed=123,
                                   variable_importances = TRUE,
                                   activation="MaxoutWithDropout",
                                   hidden = c(400,400),
                                   l1 = 1e-04,
                                   l2 = 1e-05,
                                   max_w2 = 0.1,
                                   epochs = 10,
                                   initial_weight_distribution = "Normal",
                                   balance_classes = TRUE,
                                   sparse=TRUE,
                                   reproducible = TRUE,
                                   nfolds=5)
model_11_epoch
model_11_epoch_predict <- h2o.predict(model_11_epoch,newdata = churnTest)
model_11_epoch_predict <- as.data.frame(model_11_epoch_predict)
head(model_11_epoch_predict)
confusionMatrix(model_11_epoch_predict$predict,test$Churn,positive='Yes')
#Accuracy of the model is 68.29% and sensitivity is 85.51%
#Hidden dropout ratio is added to the next model to check for improvement



model_12_epoch <- h2o.deeplearning(x=x,
                                   y=y,
                                   training_frame = churnTrain,
                                   seed=123,
                                   variable_importances = TRUE,
                                   activation="MaxoutWithDropout",
                                   hidden = c(400,400),
                                   hidden_dropout_ratios = c(0.2,0.2),
                                   l1 = 1e-04,
                                   l2 = 1e-05,
                                   max_w2 = 0.1,
                                   epochs = 10,
                                   initial_weight_distribution = "Normal",
                                   balance_classes = TRUE,
                                   sparse=TRUE,
                                   reproducible = TRUE,
                                   nfolds=5)
model_12_epoch
model_12_epoch_predict <- h2o.predict(model_12_epoch,newdata = churnTest)
model_12_epoch_predict <- as.data.frame(model_12_epoch_predict)
head(model_12_epoch_predict)
confusionMatrix(model_12_epoch_predict$predict,test$Churn,positive='Yes')
perf_12_epoch<- h2o.performance(model_12_epoch,churnTest)
h2o.auc(perf_12_epoch)

# Accuracy is 66.2% and the sensitivity is 87.15% and specificity is 58.44%
#Further tuning the model will increase the sensitivity at the cost
#of accuracy. so model_12_epoch is considered as the best model
#as the objective is to predict churn. This model is able to predict
#churn with an accuracy of 87.15%.and an AUC of 82.11




####################################################################
################### (Neural network without epoch) ####################

#Create an initial model. As most research papers and Andrew Ng
#suggests one hidden layer with number of neurons equal to 3 times
#number of inputs give good results, the same is tried and results checked

model_1 <- h2o.deeplearning(x=x,
                            y=y,
                            training_frame = churnTrain,
                            seed=123,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(90),
                            hidden_dropout_ratio = c(0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)
model_1
model_1_predict <- h2o.predict(model_1,newdata = churnTest)
model_1_predict <- as.data.frame(model_1_predict)
head(model_1_predict)
confusionMatrix(model_1_predict$predict,test$Churn,positive='Yes')
perf_1<- h2o.performance(model_1,churnTest)
h2o.auc(perf_1)
#single layer with 90 neurons.
#Train - AUC 75, logloss - 5.55, Misclassification error-27.4%
#CV - AUC 72%, logloss - 5.01, Misclassification error-29.6%
#Test - Accuracy 73.4%, senstivity 70.33, AUC - 75.2%
#The results are not very encouraging. Both the train and test errors 
#are high. This indicates that BIAS is very high.To improve BIAS complexity
#of the model is increased.
#Number of neurons is increased in the next model



model_2 <- h2o.deeplearning(x=x,
                            y=y,
                            training_frame = churnTrain,
                            seed=123,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(400),
                            hidden_dropout_ratio = c(0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)
model_2
model_2_predict <- h2o.predict(model_2,newdata = churnTest)
model_2_predict <- as.data.frame(model_2_predict)
head(model_2_predict)
confusionMatrix(model_2_predict$predict,test$Churn,positive='Yes')
perf_2 <- h2o.performance(model_2,churnTest)
h2o.auc(perf_2)
#single layer with 400 neurons.
#Train - AUC 72.9%, logloss - 7.76, Misclassification error-28.3%
#CV - AUC 74.2, logloss - 5.62, Misclassification error-25.3%
#Test - Accuracy is 73.6% and sensitivity is 62.15% only, AUC is 74.22
#The results are not very encouraging. Both the train and test errors 
#are high. This indicates that BIAS is very high.To improve BIAS complexity
#of the model is further increased.


model_3 <- h2o.deeplearning(x=x,
                            y=y,
                            training_frame = churnTrain,
                            seed=123,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(400,400,400),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)
model_3
model_3_predict <- h2o.predict(model_3,newdata = churnTest)
model_3_predict <- as.data.frame(model_3_predict)
head(model_3_predict)
confusionMatrix(model_3_predict$predict,test$Churn,positive='Yes')
perf_3 <- h2o.performance(model_3,churnTest)
h2o.auc(perf_3)
#Train AUC is 76.7% logloss is 8.15 and Misclassification error is 23.5%
#CV AUC is 68.4%, logloss is 8.35 and misclassification error is 24.2%
# Test AUC is 73.4%, Accuracy is 76.12% and sensitivity is 67.52%
#The results improved after increasing the number of layers and
#neurons also. But the sensitivity is very low. changing the Number of neurons
#in the layers and activation function to see if sensitivity increases



model_4 <- h2o.deeplearning(x=x,
                            y=y,
                            training_frame = churnTrain,
                            seed=123,
                            variable_importances = TRUE,
                            activation="RectifierWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-10,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            epoch=1,
                            reproducible = TRUE,
                            keep_cross_validation_predictions = TRUE)

model_4
model_4_predict <- h2o.predict(model_4,newdata = churnTest)
model_4_predict <- as.data.frame(model_4_predict)
head(model_4_predict)
confusionMatrix(model_4_predict$predict,test$Churn,positive='Yes')
perf_4 <- h2o.performance(model_4,churnTest)
h2o.auc(perf_4)
#Train - AUC 76.9%, logloss - 9.22, Misclassification error-27%
#CV - AUC 71, logloss - 8.59, Misclassification error-25.15%
#Test - Accuracy is 68.79% and sensitivity is 77.57%, specificity 65.54%, AUC is 70.9%
#sensitivity improved and this shows that this particular activation function
#and number of neurons are suitable for sensitivity. There is a 2% difference
#between train and validation misclassification error. So trying to reduce 
#variance and improve model performance


model_5 <- h2o.deeplearning(x=x,
                            y=y,
                            training_frame = churnTrain,
                            seed=123,
                            variable_importances = TRUE,
                            activation="RectifierWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.2,0.2,0.2),
                            l1 = 1e-10,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            epoch=1,
                            reproducible = TRUE,
                            keep_cross_validation_predictions = TRUE)

model_5
model_5_predict <- h2o.predict(model_5,newdata = churnTest)
model_5_predict <- as.data.frame(model_5_predict)
head(model_5_predict)
confusionMatrix(model_5_predict$predict,test$Churn,positive='Yes')
perf_5 <- h2o.performance(model_5,churnTest)
h2o.auc(perf_5)
#Train - AUC 77.1%, logloss - 8.62, Misclassification error-25.2%
#CV - AUC 72.94, logloss - 8.19, Misclassification error-27.31%
#Test - Accuracy is 68.67% and sensitivity is 82.94%, specificity 63.38%, AUC is 73.33%
#sensitivity and AUC improved and this shows that this we are
#in the correct direction. There is a 2% difference
#between train and validation misclassification error. So trying to further reduce 
#variance and improve model performance

model_6    <-    h2o.deeplearning(x=x,
                                   y=y,
                                   training_frame = churnTrain,
                                   seed=123,
                                   variable_importances = TRUE,
                                   activation="RectifierWithDropout",
                                   hidden = c(200,200,200),
                                   hidden_dropout_ratios = c(0.3,0.3,0.3),
                                   l1 = 1e-04,
                                   l2 = 1e-05,
                                   max_w2 = 0.1,
                                   epochs = 1,
                                   initial_weight_distribution = "Normal",
                                   balance_classes = TRUE,
                                   sparse=TRUE,
                                   reproducible = TRUE,
                                   nfolds=5)
                                   
                             
model_6
model_6_predict <- h2o.predict(model_6,newdata = churnTest)
model_6_predict <- as.data.frame(model_6_predict)
head(model_6_predict)
confusionMatrix(model_6_predict$predict,test$Churn,positive='Yes')
perf_6 <- h2o.performance(model_6,churnTest)
h2o.auc(perf_6)
#Train - AUC 84.89%, logloss - 8.22, Misclassification error-23.34%
#CV - AUC 82.3, logloss - 4.34, Misclassification error-24%
#Test - Accuracy is 71.45% and sensitivity is 85.75%, specificity 66.15%, AUC is 84.12%
#In this model both accuracy and sensitivity increased.Varaince decreased.
#As further increasing the sensitivity will adversely decrease the 
#accuracy, this model is considered as the final model without epoch


##########################################################################
#The best model without epoch is the following:-
#model_6
#Test - Accuracy is 71.45% and sensitivity is 85.75%, 
#specificity 66.15%, AUC is 84.12%
#As the objective is to predict churn, sensitivity needs to be high
#This model has got the highest senstitivity. so this is considered the best

#The best model with epoch is the following:-
#model_12_epoch
# Accuracy is 66.2% and the sensitivity is 87.15% and specificity is 58.44%
#AUC of the model is 82.11%
#As the objective is to predict churn, sensitivity needs to be high
#This model has got the highest senstitivity. so this is considered the best

#Overall best model is:-
#model_6
#Compared to two best models, one with and one without epoch,
#This model's accuracy and auc is very high with a little drop
#in sensitivity. So this is considered as the
#overall best. The metrics for the overall best model is as follows:-
#Test - Accuracy is 71.45% and sensitivity is 85.75%, 
#specificity 66.15%, AUC is 84.12%

#The variables which are important for the sake of prediction is
#saved as a dataframe in the decreasing order of importance
varimp_model_6 <- as.data.frame(h2o.varimp(model_6))

############### END OF Assignment ####################################
