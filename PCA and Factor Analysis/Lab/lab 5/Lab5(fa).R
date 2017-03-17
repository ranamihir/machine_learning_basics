###################################################################################################
# The Boston housing data. This data set was collected and analysed by Harrison and Rubinfeld (1978)
# to find out whether 'clean air ' had an influence on house prices. The data set includes
# a number of variables on neighbourhood characteristics. It comprises 506 observations for each 
# census district of the Boston metropolitan area. The variables are as follows:
# 	X1 = per capita crime rate
#	X2 = proportion of residential land zoned for large lots
#	X3 = proportion of nonretail business acres
#	X4 = Charles river (1 if tract bounds river, 0 otherwise) 
#	X5 = nitric oxides concentration
#	X6 = average number of rooms per dwelling
#	X7 = proportion of owner-occupied units built prior to 1940
#	X8 = weighted distances to 5 Boston employment centers
#	X9 = index of accessibility to radial highways
#	X10= full-value property tax rate per 10000$
#	X11= pupil/teacher ratio 
#	X12= 1000(B-0.63)^2*I(B<0.63) where B is the proportion of African American
#	X13= % lower status of the population
#	X14= median value of owner-occupied homes in 1000$
#
# We are going to study the correlation structure of these variables by factor analysis. Is it 
# possible to describe the measured variables by a smaller number of factors?
#
 data<-read.table("BostonHousing.txt",header=T)
#
# 1) Start by exploring the variables in this data set. Observe that most of the variables 
# exhibit an asymmetry, therefore it is proposed to transform the variables.  
# 
# plot(data[,1],data[,3])
xt = data
xt[,1]  = log(data[,1])
xt[,2]  = data[,2]/10
xt[,3]  = log(data[,3])
xt[,5]  = log(data[,5])
xt[,6]  = log(data[,6])
xt[,7]  = (data[,7]^(2.5))/10000
xt[,8]  = log(data[,8])
xt[,9]  = log(data[,9])
xt[,10] = log(data[,10])
xt[,11] = exp(0.4 * data[,11])/1000
xt[,12] = data[,12]/100
xt[,13] = sqrt(data[,13])
xt[,14] = log(data[,14])
#
#  2) Perform now factor analysis on this data set. Exclude the binary Charles river variable.
#     Use k = 3 factors. Estimate the factor model using the pricipal component method, principal factor method,
#     and maximum likelihood method. Use varimax rotation and interpret the factors after rotation. Summarize the
#     main characteristics for each estimated model and compare the models. Try to interpret the factors. 
#     Calculate the factor scores by regression method. Could we call the obtained factors as "quality of life factor", "employment factor"
#     and "residential factor"?
#  
data = xt[,-4]
dim(data)
dataSc = scale(data) # standardize variables
dataCor = cor(dataSc)  # correlation matrix; the same as sum(cor(data)-dataCor)
#
############# I. Estimation method: principal component method
#
e      = eigen(dataCor) # spectral decomposition of the correlation matrix
e$values                # study the eigenvalues, gives an indication of how many factors to consider
eigval = e$values[1:3]        # we consider 3 factors
eigvec = e$vectors[,1:3]
q1<-sqrt(eigval[1])*eigvec[,1]; # 1st column of the loadings matrix Q
q2<-sqrt(eigval[2])*eigvec[,2]; # 2nd column of the loadings matrix Q
q3<-sqrt(eigval[3])*eigvec[,3]; # 3rd column of the loadings matrix Q 
Q<-cbind(q1,q2,q3)
#
com<-diag(Q%*%t(Q));          # communalities are calculated
psi <- diag(dataCor) - com       # specific variances are calculated 
pcmtable<-cbind(Q,com,psi)
round(pcmtable, digits=4)
# sum(Q[,1]^2);                 # variance explained by factor1
# sum(Q[,2]^2);                 # variance explained by factor2
# sum(Q[,3]^2);                 # variance explained by factor3
# sum(Q[,1]^2)+sum(Q[,2]^2)+sum(Q[,3]^2) # so much do the 3 factors together explain of the total variation
rotQ    = varimax(Q)            # rotates the factor loadings matrix
rotLoad = rotQ$loadings         # estimated factor loadings after varimax
comrot    = diag(rotLoad %*% t(rotLoad))     # communalities with rotated loadings
psirot    = diag(dataCor) - comrot   # specific variances after rotation
pcmrot    = cbind(rotLoad,comrot, psirot) # the results from FA after VARIMAX rotation, interpret!
round(pcmrot, digits=4)
# Calculate rms = sqrt(sum of the squared off-diagonal residuals/ p(p-1))
resid<-dataCor-rotLoad %*% t(rotLoad)
sqrt(sum((resid-diag(psirot))^2)/(13*12))
# sum(rotLoad[,3]^2)
###################################################################################
# II. Estimation method: principal factor method
# We use the function 'fa' from the package 'psych'
#

pfa<-fa(dataSc,nfactors=3,fm="pa",SMC=TRUE,rotate="varimax",max.iter=20) # Which initial communalities are used?
print(pfa)
pfatable = cbind(pfa$loadings, (1 - pfa$uniquenesses), pfa$uniquenesses)
colnames(pfatable) = c("q1", "q2", "q3", "Communalities", "Specific variances")
print(round(pfatable,4))
#
pfa$rms   # the folowing 2 rows explain what it is
# resid<-dataCor-pfa$loadings%*%t(pfa$loadings)
# sqrt(sum((resid-diag(pfa$uniquenesses))^2)/(13*12))
#dim(pfa$scores); plot(pfa$scores[,1],pfa$scores[,3])
####################################################################################
# III. Estimation method: maximum likelihood method
#
# Function 'factanal' 
#
mlm  = factanal(dataSc, factors=3, rotation = "varimax")
print(mlm)
mlmtable = cbind(mlm$loadings, (1 - mlm$uniquenesses), mlm$uniquenesses)
colnames(mlmtable) = c("q1", "q2", "q3", "Communalities", "Specific variances")
print(round(mlmtable,4))
#
# Calculate RMSE
