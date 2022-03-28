setwd('C:/Users/Hp/Desktop/SL PROJECT SECOND SEM')
getwd()
#detach(bike_share)
rm(list=ls())

##Required Libraries
# data assessment/visualizations
library(readr)
library(dplyr)
library(relaimpo)
library(RColorBrewer)
library(plotly) # for plot rendering
library(lubridate)
library(date) # for handling dates
library(pacman)
library(forecast)
library(MLmetrics)
library(sqldf)
library(DT)
library(data.table)
library(pander)
library(dummies)
library(FNN) 
library(glmnet)



library(ggplot2)
library(cowplot)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)


# data wrangling
library(tidyverse)
library(forcats)
library(stringr)
library(caTools)

# model
library(xgboost)
library(MLmetrics)
library('randomForest') 
library('rpart')
library('rpart.plot')
library('car')
library('e1071')
library(vcd)
library(ROCR)
library(pROC)
library(VIM)
library(glmnet) 
library(h2o)
library(ranger)
library(broom.mixed)

#Parallel computing
library(doParallel)

#Find out how many cores are available (if you don't already know)
cores<-detectCores() ##I have 4 cores

#Create cluster with desired number of cores, leave one open for the machine        
#core processes
cl <- makeCluster(cores[1]-1) ##use 3 cores to run

#Register cluster
registerDoParallel(cl)



##Bike sharig data 
# Reading the csv files- loading the data set.
bike_share <- read.csv("hour.csv", header=T)


##SUMMARY AND STRUCTURE OF THE DATA
summary(bike_share)
attach(bike_share)


###descriptive statistics
str(bike_share) #check the structure of the data set
colnames(bike_share)
dim(bike_share) ##We have 17379  rows and 17 cols


##check for missing vaues-there are no missing values
print(paste("The total number of missing data are",sum(is.na(bike_share))))


##We can ignore column id, casual and registered coz their sum is equal to count which is our focus
bike_share <- bike_share[,-c(1,15,16)]

## we need to change the structure of season, holiday, workdday, weather
is.factor(season) ##It is not recognized as a factor

############################Data Pre-processing##########################
## convert to factor 
bike_share$season <- as.factor(season)
bike_share$holiday <- as.factor(holiday)
bike_share$workingday <- as.factor(workingday)
bike_share$weathersit <- as.factor(weathersit)
bike_share$weekday <- as.factor(weekday)


##Confirm that they are now factors
is.factor(bike_share$season) ##True
is.factor(bike_share$holiday)
is.factor(bike_share$workingday)
is.factor(bike_share$weather)

str(bike_share) ##they are now recognized as factors



##Lubridate makes it easier to do the things R does with date-times and possible to do the things R does not
##We need to convert the datetime column to actual date which are not recognized by R
##Identify the order of the year (y), month (m), day (d), hour (h), minute (m) and second (s) elements in your data. 

bike_share$dteday <- ymd(bike_share$dteday)###convert to date format
bike_share$yr <- as.factor(yr) ## 
bike_share$weekday <- as.factor(weekday)
bike_share$hr <- as.factor(hr)
bike_share$mnth <- as.factor(mnth)



str(bike_share) 

names(bike_share)

colnames(bike_share) ##we now have 14 cols





##############################  EDA  #####################################

#install.packages("gridExtra")

library(gridExtra)
bike_share$month <-month(ymd(bike_share$dteday), label=TRUE)

#1

#yr=2012
summary(bike_share[bike_share$yr=='1',])
#yr=2011
summary(bike_share[bike_share$yr=='0',])


plt4<-ggplot(bike_share, aes(fill=yr, y=cnt, x=month)) + 
  geom_bar(position="dodge", stat="identity")+
  ggtitle("Rented Bike count By Month and Year")+
  scale_fill_discrete(name='year',labels=c('2011','2012'))+
  labs(x='Month',y='Rented Bike Count')

plt4<-plt4+theme(
  legend.position = c(.05, .95),
  legend.justification = c("left", "top"),
  legend.box.just = "left",
  legend.margin = margin(3, 3, 3, 3),
  legend.key.size = unit(0.3, "cm"),
  legend.title = element_text(size = 9),
  legend.text = element_text(size = 7)
)
plt4

#2

season_summary_by_hr <- sqldf('select season, hr, avg(cnt) as count from bike_share group by season, hr')
season_summary_by_hour <- sqldf('select season, hr, avg(cnt) as count from bike_share group by season, hr')


plt1<-ggplot(bike_share, aes(x=hr, y=count, color=season))+
  geom_point(data = season_summary_by_hour, aes(group = season))+
  geom_line(data = season_summary_by_hour, aes(group = season))+
  ggtitle("Rental bike count by Hour of The Day Across Season")+ theme_minimal()+
  scale_colour_hue('Season',breaks = levels(bike_share$season), 
                   labels=c('spring', 'summer', 'fall', 'winter'))+
  labs(x='Hour of The Day',y='Rented Bike Count')

plt1<-plt1+theme(
  legend.position = c(.05, .95),
  legend.justification = c("left", "top"),
  legend.box.just = "left",
  legend.margin = margin(3, 3, 3, 3),
  legend.key.size = unit(0.3, "cm"),
  legend.title = element_text(size = 9),
  legend.text = element_text(size = 7)
)
plt1

#3  


day_summary_by_hour <- sqldf('select weekday, hr, avg(cnt) as count from bike_share group by weekday, hr')

#options(repr.plot.width = 1.5, repr.plot.height = 2)

plt3<-ggplot(bike_share, aes(x=hr, y=count, color=weekday))+
  geom_point(data = day_summary_by_hour, aes(group = weekday))+
  geom_line(data = day_summary_by_hour, aes(group = weekday))+
  ggtitle("Average Rented Bike Count by Hour of The Day across Weekdays")+ scale_colour_hue('Weekday',breaks = levels(bike_share$weekday),
                                                                                            labels=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))+
  labs(x='Hour of The Day',y='Average Rented Bike Count')

plt3<-plt3+theme(
  legend.position = c(.05, .95),
  legend.justification = c("left", "top"),
  legend.box.just = "left",
  legend.margin = margin(3, 3, 3, 3),
  legend.key.size = unit(0.3, "cm"),
  legend.title = element_text(size = 9),
  legend.text = element_text(size = 7)
)
plt3

#4


weather_summary_by_hour <- sqldf('select weathersit, hr, avg(cnt) as count from bike_share group by weathersit, hr')

#options(repr.plot.width = 1.5, repr.plot.height = 2)
plt2<-ggplot(bike_share, aes(x=hr, y=count, color=weathersit))+
  geom_point(data = weather_summary_by_hour, aes(group = weathersit))+
  geom_line(data = weather_summary_by_hour, aes(group = weathersit))+
  ggtitle("Average Rented Bike Count by Hour across Weather")+ scale_colour_hue('weathersit',breaks = levels(bike_share$weathersit),
                                                                                labels=c('Good', 'Normal', 'Bad', 'Very Bad'))+
  labs(x='Hour of The Day',y='Average Rented Bike Count')

plt2<-plt2+theme(
  legend.position = c(.05, .95),
  legend.justification = c("left", "top"),
  legend.box.just = "left",
  legend.margin = margin(3, 3, 3, 3),
  legend.key.size = unit(0.3, "cm"),
  legend.title = element_text(size = 9),
  legend.text = element_text(size = 7)
  
)
plt2

#We could put together the line Graph(the first that I see in overleaf), 
#the plot of bikes rent by SEASON, the plot of bikes rent by  WEEKDAY and 
#the plot of bikes rented by WEATHER. 

#put 2 plots in a single plot 
plot_grid(plt4,plt1, align = "v", nrow = 2)
ggsave("part1.png")

#put 2 plots in a single plot 
plot_grid(plt3,plt2, align = "v", nrow = 2)
ggsave("part2.png")

##########################residuals###########################################

#LINEAR MODEL 
par(mfrow=c(2,2))
plot(fitted(lm1), residuals(lm1), xlab="Fitted values", ylab="Residuals",main='Response Y')
lines(loess.smooth(fitted(lm1), residuals(lm1)), col="blue", lwd=2)
abline(h=0, lty=2)

plot(fitted(lm3), residuals(lm3), xlab="Fitted values", ylab="Rresiduals",main='Response log(Y)')
lines(loess.smooth(fitted(lm3), residuals(lm3)), col="blue", lwd=2)
abline(h=0, lty=2)

qqnorm(residuals(lm1),main='Response Y')
qqline(residuals(lm1))
qqnorm(residuals(lm3),main='Response log(Y)')
qqline(residuals(lm3))

#POISSON MODEL

par(mfrow=c(2,2))
plot(fitted(glm.fit), residuals(glm.fit), xlab="Fitted values", ylab="Residuals",main='Response Y')
lines(loess.smooth(fitted(glm.fit), residuals(glm.fit)), col="blue", lwd=2)
abline(h=0, lty=2)

plot(fitted(glm.fit3), residuals(glm.fit3), xlab="Fitted values", ylab="Rresiduals",main='Response log(Y)')
lines(loess.smooth(fitted(glm.fit3), residuals(glm.fit3)), col="blue", lwd=2)
abline(h=0, lty=2)

qqnorm(residuals(lm1),main='Response Y')
qqline(residuals(lm1))
qqnorm(residuals(lm3),main='Response log(Y)')
qqline(residuals(lm3))


######################################################
bike_share <- bike_share[, -1] ##remove the dteday for analysis

##############Random Forest training##########################
# Splitting the Train dataset
set.seed(123)
split <- sample.split(bike_share$cnt, SplitRatio = 0.70) ## train 70%, test 30%
##length = 17,379 -whole data set
training_set <- subset(bike_share, split == TRUE) ##train set
## train 12,185 by 14 variables
test_set <- subset(bike_share, split == FALSE) ##test set
## test 5,194 by 14 variables


##Split the train_set to train and validation

set.seed(123)
split_2 <- sample.split(training_set$cnt, SplitRatio = 0.8) ##total trainset=12185
new_train <- subset(training_set, split_2 == TRUE)#new train = 9781
valid_set <- subset(training_set, split_2 == FALSE) ##valid = 2404
  
## train the model- RF
##We use the initial training set at first
##This takes long to load
m1 <- randomForest(
  formula = cnt ~ .,
  data    = training_set
)

m1
plot(m1)

# number of trees with lowest MSE- lowest error rate
which.min(m1$mse) ##500

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)]) ##49.29001



##Rf- predicting accuracy using validation set
# validation data

x_test <- valid_set[setdiff(names(valid_set), "cnt")] ##all predictors of the valid set
y_test <- valid_set$cnt

rf_oob_comp <- randomForest(
  formula = cnt ~ .,
  data    = new_train,
  xtest   = x_test,
  ytest   = y_test
)


# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = comma) +
  xlab("Number of trees")  

##Tunning the model
##tune the mtry parameter starting from mtry=5 to obtain optimal value of OOB error
# names of features of the initial train set
features <- setdiff(names(training_set), "cnt")
#13 variables

set.seed(123)

##start with mtry=5, with ntrees=500, stepfunct-at each iteration, mtry is incresed/decreased
##THis takes a long time to load
m2 <- tuneRF(
  x = training_set[features],
  y = training_set$cnt,
  ntreeTry   = 500,
  mtryStart  = 4,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE  ##so as not to print the progress of the search 

)
##(results)
#-0.1535296 0.01 
#0.1184505 0.01 
#0.05301279 0.01 
#-0.01049833 0.01 

#-0.1249427 0.01 
#0.08734034 0.01 
#0.02233379 0.01 
#-0.01288931 0.01 


##Ranger Implementation
# randomForest speed compared to ranger(faster)
system.time(
  bikerent_randomForest <- randomForest(
    formula = cnt ~ ., 
    data    = training_set, 
    ntree   = 500,
    mtry    = floor(length(features) / 3)
  )
)

#user  system elapsed 
#695.44    0.44  702.09 


# ranger speed
system.time(
  bikerent_ranger <- ranger(
    formula   = cnt ~ ., 
    data      = training_set, 
    num.trees = 500,
    mtry      = floor(length(features) / 3)
  )
)
#user  system elapsed 
#17.83    0.54    5.95 


# hyperparameters for ranger grid search

hyper_grid <- expand.grid(
  mtry       = seq(4, 12, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# total number of combinations = 80
nrow(hyper_grid)

### We loop through each hyperparameter combination and apply 500 trees
##
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = cnt ~ ., 
    data            = training_set, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)
#mtry node_size sampe_size OOB_RMSE
#1    10         3      0.800 42.29460
#2     8         3      0.800 42.44580
#3    10         3      0.700 42.48087
#4    10         5      0.800 42.55136
#5    10         5      0.700 42.62161
#6     8         5      0.800 42.70942
#7    10         7      0.800 42.79754
#8    10         3      0.632 42.80450
#9     8         3      0.700 42.84049
#10   10         5      0.632 42.97529

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = cnt ~ ., 
    data            = training_set, 
    num.trees       = 500,
    mtry            = 10,
    min.node.size   = 5,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

##Rank of variables according to importance
library(broom.mixed)
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(13) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top variables")


optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(4) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 4 important variables")


####perform one hot encoding to check if we get better performance

# one-hot encode our categorical variables
one_hot <- dummyVars(~ ., training_set, fullRank = FALSE)
Bikeshare_train_hot <- predict(one_hot, training_set) %>% as.data.frame()

# make ranger compatible names
names(Bikeshare_train_hot) <- make.names(names(Bikeshare_train_hot), allow_ = FALSE)

# hyperparameter grid search --> same as above but with increased mtry values
hyper_grid_2 <- expand.grid(
  mtry       = seq(4, 12, by = 1),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE  = 0
)

# perform grid search
for(i in 1:nrow(hyper_grid_2)) {
  
  # train model
  model <- ranger(
    formula         = cnt ~ ., 
    data            = Bikeshare_train_hot, 
    num.trees       = 500,
    mtry            = hyper_grid_2$mtry[i],
    min.node.size   = hyper_grid_2$node_size[i],
    sample.fraction = hyper_grid_2$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid_2$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid_2 %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

#mtry node_size sampe_size OOB_RMSE
#1    12         3      0.800 55.34074
#2    12         5      0.800 55.89757
#3    11         3      0.800 55.91597
#4    12         3      0.700 56.04713
#5    12         7      0.800 56.33357
#6    12         5      0.700 56.47617
#7    11         3      0.700 56.56555
#8    12         3      0.632 56.68482
#9    12         9      0.800 56.74901
#10   11         5      0.800 56.77955



##this is where ill start tommorow
##perform a h2o grid search from java which is much faster
# start up h2o (I turn off progress bars when creating reports/tutorials)
set.seed(123)
#Sys.setenv(JAVA_HOME="C:/Users/Hp/Downloads/openjdk-13+33_windows-x64_bin/jdk-13")
h2o.no_progress()
#h2o.init(startH2O = FALSE)
h2o.init(max_mem_size = "5g")
##  Connection successful!


# create feature names
y <- "cnt"
x <- setdiff(names(training_set), y)

# turn training set into h2o object
train.h2o <- as.h2o(training_set)

h2o_rf_1 <- h2o.randomForest(x = x,
                             y= y,
                             training_frame = train.h2o,
                             ntrees = 13 * 10,
                             seed = 123
                             )

h2o_rf_1


# hyperparameter grid
hyper_grid.h2o <- list(
  mtries      = c(0.65, 1.5, 3.25, 5 ),
  min_rows    = c(1, 3, 5, 10),
  max_depth   = c(10, 20, 30),
  sample_rate = c(.55, .632, .70, .80)
)

##random grid search strategy
search_criteria_1 <- list (
  strategy = 'RandomDiscrete',
  stopping_metric = 'mse',
  stopping_tolerance = 0.001,
  stopping_rounds = 10,
  max_runtime_secs = 60 * 30
)



# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid",
  x = x, 
  y = y, 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  ntrees = 130,
  seed = 123,
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  search_criteria = search_criteria_1
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "rf_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)


# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now lets evaluate the model performance on a test set
Bikeshare_test.h2o <- as.h2o(test_set)
best_model_perf <- h2o.performance(model = best_model, newdata = Bikeshare_test.h2o)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()
##42.6077

##Predicting
pred_rf <- predict(bikerent_randomForest, test_set)
head(pred_rf, 10)

pred_ranger<- predict(bikerent_ranger, test_set)
head(pred_ranger$predictions, 10)


pred_h2o<- predict(best_model, Bikeshare_test.h2o)
head(pred_h2o, 10)


head(test_set$cnt,10)

#####################################################################################

##################### LINEAR MODEL##################################################

set.seed(123)
split <- sample.split(bike_share$cnt, SplitRatio = 0.70)
train_set <- subset(bike_share, split == TRUE)
test_set <- subset(bike_share, split == FALSE)
test.values<-test_set$cnt

#######LM all variables (except for workingday)

lm1 <- lm(cnt~.-workingday, data = train_set)
summary(lm1)

y.pred<-predict(lm1,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(2,2))
plot(lm1)

########LM all variables 10 CV(except for workingday)
set.seed(123) 
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model1 <- train(cnt~ .-workingday, data = train_set, method = "lm",
                trControl = train.control)
print(model1)

y_pred <- predict(model1, newdata = test_set)

postResample(y_pred,test_set$cnt)

##########LM only categorical variables

lm2<-lm(cnt~season+holiday+weekday+weathersit+yr+mnth+hr,data=train_set)
summary(lm2)

y.pred<-predict(lm2,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(lm2)

##########LM only categorical variables 10 CV
set.seed(123) 
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model2 <- train(cnt~ season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "lm",
                trControl = train.control)
print(model2)

y_pred <- predict(model2, newdata = test_set)

postResample(y_pred,test_set$cnt)


###########Log transformation of response

#######LM all variables with logY(except for workingday)

lm3 <- lm(log(cnt)~.-workingday, data = train_set)
summary(lm3)

y.pred<-predict(lm3,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(lm3)

########LM all variables 10 CV  with logY(except for workingday)
set.seed(123) 
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model3 <- train(log(cnt)~ .-workingday, data = train_set, method = "lm",
                trControl = train.control)
print(model3)

y_pred <- predict(model3, newdata = test_set)

postResample(exp(y_pred),test_set$cnt)

##########LM only categorical variables with logY

lm4<-lm(log(cnt)~season+holiday+weekday+weathersit+yr+mnth+hr,data=train_set)
summary(lm4)

y.pred<-predict(lm4,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(lm4)

##########LM only categorical variables 10 CV
set.seed(123) 
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model4 <- train(log(cnt)~ season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "lm",
                trControl = train.control)
print(model4)

y_pred <- predict(model4, newdata = test_set)

postResample(exp(y_pred),test_set$cnt)

###############################GLM with Lasso or elasticnet regularization##################################################

#########Regularized LM with logY and all variables 10 CV (except for workingday)

set.seed(123)
#k cross validation
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
# Train the model
modelr1 <- train(log(cnt)~.-workingday, data = train_set, method = "glmnet",family='gaussian',
                 trControl = train.control)

print(modelr1)

y_pred <- predict(modelr1, newdata = test_set)

postResample(exp(y_pred),test_set$cnt)

##########LM only categorical variables with logY

set.seed(123)
#k cross validation
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
# Train the model
modelr1 <- train(log(cnt)~season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "glmnet",family='gaussian',
                 trControl = train.control)

print(modelr1)

y_pred <- predict(modelr1, newdata = test_set)

postResample(exp(y_pred),test_set$cnt)



#############################POISSON MODEL###############################################

#######PM all variables(except for workingday)

glm.fit<-glm(cnt~.-workingday,data=train_set,family = poisson)
summary(glm.fit)

y.pred<-predict(glm.fit,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(glm.fit)

########PM all variables 10 CV(except for workingday)

set.seed(123)
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model5 <- train(cnt~.-workingday, data = train_set, method = "glm",family="poisson",
                trControl = train.control)

print(model5)
y_pred <- predict(model5, newdata = test_set)
postResample(y_pred,test_set$cnt)

##########LM only categorical variables

glm.fit2<-glm(cnt~season+holiday+weekday+weathersit+yr+mnth+hr,data=train_set,family = poisson)
summary(glm.fit2)

y.pred<-predict(glm.fit2,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(glm.fit2)

##########LM only categorical variables 10 CV

set.seed(123)
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model6 <- train(cnt~season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "glm",family="poisson",
                trControl = train.control)

print(model6)
y_pred <- predict(model6, newdata = test_set)
postResample(y_pred,test_set$cnt)

##########################Log transformation of response

#######PM all variables with logY(except for workingday)

glm.fit3<-glm(log(cnt)~.-workingday,data=train_set,family = poisson)
summary(glm.fit3)

y.pred<-predict(glm.fit3,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(1,2))
plot(glm.fit3)

########PM all variables 10 CV  with logY(except for workingday)

set.seed(123)
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model7 <- train(log(cnt)~.-workingday, data = train_set, method = "glm",family="poisson",
                trControl = train.control)

print(model7)
y_pred <- predict(model7, newdata = test_set)
postResample(exp(y_pred),test_set$cnt)

##########PM only categorical variables  with logY

glm.fit4<-glm(log(cnt)~season+holiday+weekday+weathersit+yr+mnth+hr,data=train_set,family = poisson)
summary(glm.fit4)

y.pred<-predict(glm.fit4,newdata = test_set)
rmse.lm = rmse(test_set$cnt, y.pred)
mae.lm=mae(test_set$cnt, y.pred)
print(paste("RMSE: ",rmse.lm))
print(paste("MAE: ",mae.lm))

par(mfrow=c(2,2))
plot(glm.fit4)

##########PM only categorical variables 10 CV  with logY

set.seed(123)
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model8 <- train(log(cnt)~season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "glm",family="poisson",
                trControl = train.control)

print(model8)
y_pred <- predict(model8, newdata = test_set)
postResample(exp(y_pred),test_set$cnt)

################################Regularized Poisson Model################################

#####################with all variables

set.seed(123)

train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model <- train(cnt~.-workingday, data = train_set, method = "glmnet",family="poisson",
               trControl = train.control)

print(model)
y_pred <- predict(model, newdata = test_set)

postResample(y_pred,test_set$cnt)

#####################with only categorical variables

set.seed(123)

train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
model <- train(log(cnt)~season+holiday+weekday+weathersit+yr+mnth+hr, data = train_set, method = "glmnet",family="poisson",
               trControl = train.control)

print(model)
y_pred <- predict(model, newdata = test_set)

postResample(y_pred,test_set$cnt)



###################################################################################

#### Predictive Models Analysis NO NUMERICAL VARS M1####
bike_share_M1 <- bike_share[,-c(1,10,11,12,13)] 

#dummy code variables that are factors
bike_share.new1 <- dummy.data.frame(bike_share_M1, sep = ".")


# Splitting into train(70%) and test(30%) sets
set.seed(123)
split <- sample.split(bike_share.new1$cnt, SplitRatio = 0.7)
training_set_M1 <- subset(bike_share.new1, split == TRUE)
test_set_M1 <- subset(bike_share.new1, split == FALSE)

#Selecting feature Matrix X and target Variable Y
X_train_M1= training_set_M1[,-c(58)]
X_test_M1 = test_set_M1[,-c(58)]
y_train_M1 = training_set_M1["cnt"]
y_test_M1 = test_set_M1["cnt"]

#KNN Model for different values of K 
#pred_knn1_M1=knn.reg(train =X_train_M1, test =X_test_M1 , y = y_train_M1, k = 1)
pred_knn5_M1=knn.reg(train =X_train_M1, test =X_test_M1 , y = y_train_M1, k = 5)
#pred_knn10=knn.reg(train =X_train_M1, test =X_test_M1 , y = y_train_M1, k = 10)


#Now check for the RMSE, RQquared and MAE values
#Which are the most famous measures of performance in regression models
#https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d

#printing RMSE R² and MAE
postResample(pred = pred_knn5_M1$pred, obs = t(y_test_M1))



#### Predictive Models Analysis NUM VARS  M2 ----
bike_share_M2 <- bike_share[,-c(1)]

#dummy code variables that are factors
bike_share.new2 <- dummy.data.frame(bike_share_M2, sep = ".")


# Splitting into train(70%) and test(30%) sets
set.seed(123)
split <- sample.split(bike_share.new2$cnt, SplitRatio = 0.7)
training_set_M2 <- subset(bike_share.new2, split == TRUE)
test_set_M2 <- subset(bike_share.new2, split == FALSE)

#Selecting feature Matrix X and target Variable Y
X_train_M2= training_set_M2[,-c(62)]
X_test_M2 = test_set_M2[,-c(62)]
y_train_M2 = training_set_M2["cnt"]
y_test_M2 = test_set_M2["cnt"]

#KNN Model for different values of K 
# pred_knn1_M2=knn.reg(train =X_train_M2, test =X_test_M2 , y = y_train_M2, k = 1)
pred_knn5_M2=knn.reg(train =X_train_M2, test =X_test_M2 , y = y_train_M2, k = 5)
#pred_knn10=knn.reg(train =X_train, test =X_test , y = y_train, k = 10)

#printing RMSE R² and MAE
postResample(pred = pred_knn5_M2$pred, obs = t(y_test_M2))


##Using Cross_Validation M2####
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

knn_fit <- train(cnt~., data = training_set_M2, method = "knn",
                 trControl=trctrl,tuneLength = 10)


test_pred_M2 <- predict(knn_fit, newdata = test_set_M2)
test_pred_M2

MAE(test_pred_M2, t(y_test_M2))
RMSE(test_pred_M2, t(y_test_M2))

#Using CV for M2: RMSE=103.6 MAE=71.77
#Using Cross_Validation M1####
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

knn_fit <- train(cnt~., data = training_set_M1, method = "knn",
                 trControl=trctrl,tuneLength = 10)


test_pred_M1 <- predict(knn_fit, newdata = test_set_M1)
# test_pred_M1

MAE(test_pred_M1, t(y_test_M1))
RMSE(test_pred_M1, t(y_test_M1))

#Using CV for M1: RMSE=129.96 MAE=98.008

#Maybe we should scale the data for the predicted variable 







#XGboost for Regression M2----


#Training with xgboost - gives better scores than 'rf'
trctrl <- trainControl(method = "cv", number = 10)

# Takes a long to time to run in kaggle gridsearch
#tune_grid <- expand.grid(nrounds=c(100,200,300,400), 
#                         max_depth = c(3:7),
#                         eta = c(0.05, 1),
#                         gamma = c(0.01),
#                         colsample_bytree = c(0.75),
#                         subsample = c(0.50),
#                         min_child_weight = c(0))

# Tested the above setting in local machine
tune_grid <- expand.grid(nrounds = 200,
                         max_depth = 5,
                         eta = 0.05,
                         gamma = 0.01,
                         colsample_bytree = 0.75,
                         min_child_weight = 0,
                         subsample = 0.5)

rf_fitM2 <- train(cnt~., data = training_set_M2, method = "xgbTree",
                  trControl=trctrl,
                  tuneGrid = tune_grid,
                  tuneLength = 10)
# have a look at the model 
rf_fitM2

# Testing
test_pred_M2 <- predict(rf_fitM2, newdata = test_set_M2)
# test_pred_M2

MAE(test_pred_M2, t(y_test_M2))
RMSE(test_pred_M2, t(y_test_M2))


#RMSE= 57.91487 and MAE=40.44154 Using all the variables




#XGboost for Regression M1----


rf_fitM1 <- train(cnt~., data = training_set_M1, method = "xgbTree",
                  trControl=trctrl,
                  tuneGrid = tune_grid,
                  tuneLength = 10)
# have a look at the model 
rf_fitM1

# Testing
test_pred_M1 <- predict(rf_fitM1, newdata = test_set_M1)
# test_pred_M1

MAE(test_pred_M1, t(y_test_M1))
RMSE(test_pred_M1, t(y_test_M1))

# RMSE: 64.25624 and MAE: 44.5345





