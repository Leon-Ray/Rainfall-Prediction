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
train = train[Ref != "NA", ]

#This simple outlier removal method resulted in a lower MAE in the test data than more sophisticated methods that I tried
q3 = quantile(train$Expected, .75)
iqr = IQR(train$Expected)
lim = q3 + iqr
train = train[Expected < lim, ]

rm(q3, iqr, lim)


###########################
### Process Train Data  ###
###########################

#Generate features for model development
train2 = train[,list(observations=.N, radardist_km=mean(radardist_km), 
                     mean.ref=mean(Ref), median.ref=median(Ref),SD.ref=sd(Ref), min.ref=min(Ref), max.ref=max(Ref),
                     mean.refcomp=mean(RefComposite), median.refcomp=median(RefComposite),SD.refcomp=sd(RefComposite), min.refcomp=min(RefComposite), max.refcomp=max(RefComposite),
                     expected=mean(Expected)), 
               by=Id]

#Set 0 for standard deviations that are undefined due to one observation (R uses denominator N-1)
train2$SD.ref[is.na(train2$SD.ref)] = 0
train2$SD.refcomp[is.na(train2$SD.refcomp)] = 0

default = median(train2$expected)


#########################
### Fit Linear Model  ###
#########################

#Perform stepwise linear regression for feature selection
full=lm(expected~observations+radardist_km+mean.ref+median.ref+SD.ref+min.ref+max.ref+mean.refcomp+median.refcomp+SD.refcomp+min.refcomp+max.refcomp, data = train2)
null=lm(expected~1,data=train2)
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
test = test[Ref != "NA", ]

#Generate features for model application
test2 = test[,list(observations=.N, radardist_km=mean(radardist_km), 
                   mean.ref=mean(Ref), median.ref=median(Ref),SD.ref=sd(Ref), min.ref=min(Ref), max.ref=max(Ref),
                   mean.refcomp=mean(RefComposite), median.refcomp=median(RefComposite),SD.refcomp=sd(RefComposite), min.refcomp=min(RefComposite), max.refcomp=max(RefComposite)), 
             by=Id]

#Set 0 for standard deviations that are undefined due to one observation (R uses denominator N-1)
test2$SD.ref[is.na(test2$SD.ref)] = 0
test2$SD.refcomp[is.na(test2$SD.refcomp)] = 0


#################################
### Apply Model to Test Data  ###
#################################

#Generalize model application
test2[,"(Intercept)" := 1] #Dummy "variable" for intercept "coefficient"
test2 = test2[, colnames(test2) %in% c(names), with = FALSE] #Only include modeled features
test2 = test2[, c(names), with = FALSE] #Reorder variables to match model coefficients
test2[, Expected := as.matrix(test2[.I]) %*% coefficients] #Perform dot product of variables and coefficients


#######################################
### Post-Process and Write Results  ###
#######################################

test2$Expected[test2$Expected < 0] = 0 #Negative rain is not possible

#Create new dataframe with Id and prediction only
output = test2[, .(Id, Expected)]
setkey(output, Id)

#Where explanatory variables were not available, apply the median result from train data
Id = c(1:717625) #number of observations in the test data
all = as.data.table(Id)
setkey(all, Id)
output = merge(all, output, all.x=TRUE)
output$Expected[is.na(output$Expected)] = default

#Write submission file
write.csv(output, "submission.csv", row.names = FALSE)

rm(Id, all)
