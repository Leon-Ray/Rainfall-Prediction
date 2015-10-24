setwd("/Users/leonraykin/Kaggle/Rainfall Prediction/")
options(warn=1) #print warnings as they occur

rm(list=ls())
gc()

require("data.table")


#########################
### Read Train Data   ###
#########################

train = fread("train.csv", stringsAsFactors = F )


########################
### Remove Outliers  ###
########################

#Remove rows with missing explanatory data
train_clean = train[Ref != "NA", ]

#This simple outlier removal method resulted in a lower MAE in the test data than more sophisticated methods that I tried
q3 = quantile(train$Expected, .75)
iqr = IQR(train$Expected)
lim = q3 + iqr
train_clean = train_clean[Expected < lim, ]

rm(q3, iqr, lim)


###########################
### Process Train Data  ###
###########################

#Collapse to one record per Id and generate features for model development
train_collapsed = train_clean[,list(observations=.N, radardist_km=mean(radardist_km), 
                     mean.ref=mean(Ref), median.ref=median(Ref),SD.ref=sd(Ref), min.ref=min(Ref), max.ref=max(Ref),
                     mean.refcomp=mean(RefComposite), median.refcomp=median(RefComposite),SD.refcomp=sd(RefComposite), min.refcomp=min(RefComposite), max.refcomp=max(RefComposite),
                     expected=mean(Expected)), 
               by=Id]

#Set 0 for standard deviations that are undefined due to one observation (R uses denominator N-1)
train_collapsed$SD.ref[is.na(train_collapsed$SD.ref)] = 0
train_collapsed$SD.refcomp[is.na(train_collapsed$SD.refcomp)] = 0

default_expected = median(train_collapsed$expected)


#########################
### Fit Linear Model  ###
#########################

#Perform stepwise linear regression for feature selection
full=lm(expected~observations+radardist_km+mean.ref+median.ref+SD.ref+min.ref+max.ref+mean.refcomp+median.refcomp+SD.refcomp+min.refcomp+max.refcomp, data = train_collapsed)
null=lm(expected~1,data=train_collapsed)
step=step(null,scope=list(lower=null,upper=full),direction="forward")
fit=lm(step$model)

#Model coefficents
coefficients = fit$coefficients
coefficients["Id"] =  0 #Exclude Id from model application
names = names(coefficients)

rm(full, null, step)
gc()


########################
### Read Test Data   ###
########################

test = fread("test.csv", stringsAsFactors = F )


##########################
### Process Test Data  ###
##########################

#Remove rows with missing explanatory data
test_clean = test[Ref != "NA", ]

#Collapse to one record per Id and generate features for model application
test_collapsed = test_clean[,list(observations=.N, radardist_km=mean(radardist_km), 
                   mean.ref=mean(Ref), median.ref=median(Ref),SD.ref=sd(Ref), min.ref=min(Ref), max.ref=max(Ref),
                   mean.refcomp=mean(RefComposite), median.refcomp=median(RefComposite),SD.refcomp=sd(RefComposite), min.refcomp=min(RefComposite), max.refcomp=max(RefComposite)), 
             by=Id]

#Set 0 for standard deviations that are undefined due to one observation (R uses denominator N-1)
test_collapsed$SD.ref[is.na(test_collapsed$SD.ref)] = 0
test_collapsed$SD.refcomp[is.na(test_collapsed$SD.refcomp)] = 0


#################################
### Apply Model to Test Data  ###
#################################

#Generalize model application
test_predicted = test_collapsed
test_predicted[,"(Intercept)" := 1] #Dummy "variable" for intercept "coefficient"
test_predicted = test_predicted[, colnames(test_predicted) %in% c(names), with = FALSE] #Only include modeled features
test_predicted = test_predicted[, c(names), with = FALSE] #Reorder variables to match model coefficients
test_predicted[, Expected := as.matrix(test_predicted[.I]) %*% coefficients] #Perform dot product of variables and coefficients


#######################################
### Post-Process and Write Results  ###
#######################################

test_predicted$Expected[test_predicted$Expected < 0] = 0 #Negative rain is not possible

#Create new dataframe with Id and prediction only
output = test_predicted[, .(Id, Expected)]
setkey(output, Id)

#Where explanatory variables were not available, apply the median result from train data
Id = c(1:717625) #number of Ids in the test data
all = as.data.table(Id)
setkey(all, Id)
output = merge(all, output, all.x=TRUE)
output$Expected[is.na(output$Expected)] = default_expected

#Write submission file
write.csv(output, "submission.csv", row.names = FALSE)

rm(Id, all)
