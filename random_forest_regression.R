# UMichigan Big Data Summer Institute
# EHR Final Project
# From Interview Questions to Exam Results : A Linear Model Versus a Machine Learned Model
# Ryan Masson 

setwd("~/Documents/Extras/Data Science/UM BDSI /EHR /Project_predicting_eGFR")
library(caret)
library(randomForest)


##### data setup from Priyanka to be able to directly compare performance between models (linear and randomForest)####

# creating subset of data 2001-2014 because pre-2001 used different weighting system
data0 <- read.csv("NHANES.csv")
NHANES <- data0[data0$SDDSRVYR >= 2 & data0$SDDSRVYR <= 8, ]
NHANES$WTMEC14YR <- NHANES$WTMEC2YR/7

# Bootstrapping; sampling according to weights to mimic SRS
NHANES$weights <- (NHANES$WTMEC14YR)/sum(NHANES$WTMEC14YR)
set.seed(711)
DATA <- NHANES[sample(1:nrow(NHANES), size = 20000, replace=TRUE, prob = NHANES$weights), ]

# Prepping data for cross validation by creating 10 folds
set.seed(711)
DATA$row.number <- 1:nrow(DATA)
folds <- createFolds(1:nrow(DATA), k = 10, list = TRUE, returnTrain= FALSE)
fold.number <- vector()
for(i in 1:10){
  fold.number[folds[[i]]] <- i
}
DATA <- cbind(DATA, fold.number)

# imputing data into NAs because the randomForest package can't handle missing data
imputed_rep_sample <- na.roughfix(DATA)

########## RANDOM FOREST OF REGRESSION TREES TO PREDICT eGFR FROM LIFESTYLE/INTERVIEW VARIABLES ################

# model fitting with all variables to get importance rankings of all NHANES vars
eGFR_forest <- randomForest(y = imputed_rep_sample$CKD_epi_eGFR,
                            x = imputed_rep_sample[, -77],
                            data = imputed_rep_sample,
                            mtry = 25,
                            ntree = 50
)
# predictor importance analysis
importances <- as.data.frame(importance(eGFR_forest))
importances$name <- rownames(importances)
rownames(importances) <- 1:nrow(importances)
importances <- importances[order(importances$IncNodePurity, decreasing = TRUE),]
importances

# model fitting with just selected interview variables for importance plot on poster

lifestyle_predictor_forest <- randomForest(CKD_epi_eGFR ~ CHF_self + CHD_self + Diab_self + HTN_self + MI_self +
                                             angina_self + stroke_self + age_months + age_years + race_eth +
                                             sex + white + black + Smoke_now + insured + private_ins + Medicare_ins +
                                             Other_gov_ins + sleep_amount + vigorous_activity + meals_not_home +
                                             live_births + education + Chol_self + vigorous_rec,
                                           data = imputed_rep_sample,
                                           mtry = 12,
                                           ntree = 50
)
varImpPlot(lifestyle_predictor_forest, main = "Interview Variable Importances", pch = 19, lcolor = "purple")


# MODEL FITTING OVERVIEW:
# fit a forest and the run performance code to evaluate each model

# performance code:
predictions <- predict(eGFR_forest, newdata = imputed_rep_sample[15001:20000, ])
true_eGFR <- imputed_rep_sample[15001:20000, names(imputed_rep_sample) == "CKD_epi_eGFR"]
square_error <- as.matrix((predictions - true_eGFR)^2)
root_MSE <- sqrt(mean(square_error))
root_MSE

# MODEL 1: initial, arbitrary selection by me
eGFR_forest <- randomForest(CKD_epi_eGFR ~ sleep_amount + mod_activity + vigorous_activity + vigorous_rec +
                              Smoking + insured, 
                            data = imputed_rep_sample[1:15000, ], mtry = 2, ntree = 1000
)
# MODEL 2: top ten variables by importance ranking 
# (VERSION 1: mtry=3, ntree=500 on sample size of 20,000) (THIS IS THE MODEL WE'RE ACTUALLY PRESENTING)
eGFR_forest <- randomForest(CKD_epi_eGFR ~ age_years + Medicare_ins + BMI + HTN_self + education + white +
                              vigorous_activity + meals_not_home + sleep_amount + Smoke_now,
                            data = imputed_rep_sample[1:15000, ], mtry = 3, ntree = 500
)
# MODEL 3: top ten variables by importance ranking 
# (VERSION 2: mtry=25, ntree=50 on random sample size of 40,000; this sample size broke Priyanka's linear model)
eGFR_forest <- randomForest(CKD_epi_eGFR ~ age_years + Medicare_ins + male + black + BMI + vigorous_activity +
                              live_births + education + meals_not_home + sleep_amount,
                            data = imputed_rep_sample[1:15000, ], mtry = 3, ntree = 500
)
# MODEL 4 : top ten variables by importance ranking
# (VERSION 3: mtry=25, ntree=50 on random sample size of 20,000; see it implemented in cross validation code)



# CROSS VALIDATION OF MODELS (modified from Priyanka's original code):

data0 <- read.csv("NHANES.csv")
NHANES <- data0[data0$SDDSRVYR >= 2 & data0$SDDSRVYR <= 8, ]
NHANES$WTMEC14YR <- NHANES$WTMEC2YR/7
NHANES$weights <- (NHANES$WTMEC14YR)/sum(NHANES$WTMEC14YR)
set.seed(10)
DATA <- NHANES[sample(1:nrow(NHANES), size = 5000, replace=TRUE, prob = NHANES$weights), ]
set.seed(10)
DATA$row.number <- 1:nrow(DATA)
folds <- createFolds(1:nrow(DATA), k = 10, list = TRUE, returnTrain= FALSE)
fold.number <- vector()
for(i in 1:10){
  fold.number[folds[[i]]] <- i
}
DATA <- cbind(DATA, fold.number)
imputed_rep_sample <- na.roughfix(DATA)

for (j in 1:4) {
  
  models <- list()
  predictions <- list()
  MSE <- list()
  root_MSE <- list()
  
  for(i in 1:10){
    
    train <- imputed_rep_sample[imputed_rep_sample$fold.number != i, ]
    test <- imputed_rep_sample[imputed_rep_sample$fold.number == i, ]
    
    if(j == 1) {
      models[[i]] <- randomForest(CKD_epi_eGFR ~ sleep_amount + mod_activity + vigorous_activity + vigorous_rec +
                                    Smoking + insured, 
                                  data = train, mtry = 1, ntree = 1
      )}
    if(j == 2) {    # dis is de final model teehee
      models[[i]] <- randomForest(CKD_epi_eGFR ~ age_years + Medicare_ins + BMI + HTN_self + education + white +
                                    vigorous_activity + meals_not_home + sleep_amount + Smoke_now,
                                  data = train, mtry = 3, ntree = 400
      )}
    if(j == 3) {
      models[[i]] <- randomForest(CKD_epi_eGFR ~ age_years + Medicare_ins + male + black + BMI + vigorous_activity +
                                    live_births + education + meals_not_home + sleep_amount,
                                  data = train, mtry = 1, ntree = 1
      )}
    if(j == 4) {
      models[[i]] <- randomForest(CKD_epi_eGFR ~ age_years + Medicare_ins + male + black + Chol_self +
                                    vigorous_activity + BMI + education + Smoke_now + sleep_amount,
                                  data = train, mtry = 1, ntree = 1
      )}
    predictions <- predict(models[[i]], newdata = test)
    true_eGFR <- test[ , names(test) == "CKD_epi_eGFR"]
    MSE[[i]] <- (mean(as.matrix((predictions - true_eGFR)^2)))
    root_MSE[[i]] <- sqrt((mean(as.matrix((predictions - true_eGFR)^2))))
  }
  print(mean(as.numeric(MSE)))
  print(mean(as.numeric(root_MSE)))
}

# these are the MSE and root_MSE for runs with different random seeds (1-10)

MSEs <- c(209.3449, 205.7854, 194.4179, 203.8263, 198.8773, 
          218.1915, 202.4304, 215.1052, 200.0615, 189.0452)
root_MSEs <- c(14.43855, 14.33644, 13.93313, 14.25737, 14.08979, 
               14.75067, 14.21459, 14.64426, 14.12564, 13.73097)
mean(MSEs)
mean(root_MSEs)




# write some cheeky black box function that predicts a patient's eGFR?

