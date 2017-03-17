############################################################################################
# Example 1. Olympic decathlon data from 1988. 
# Perform factor analysis (FA) on the decathlon data. We start with removing one observation 
# which is very different from the rest. Think also about the scales of the variables, 
# so that 'large' values correspond to a 'better' result. 
# 
decathlon<-read.table("olympicDec.txt",header=T)
#fix(decathlon)
dim(decathlon)
# Remove the last object and form a data set with only 10 sports events
subDec<-decathlon[-34,2:11];
# fix(subDec)
# New variables for the running events
subDec$run100<--subDec$run100; subDec$run400<--subDec$run400; 
subDec$hurdle<--subDec$hurdle; subDec$run1500<--subDec$run1500;
##########################################################################################
# Perform now FA using the method of principal components (PCM).
# Use k = 3 factors. 
# 1) Start with the spectral decomposition of the correlation matrix R. 
#    Study the eigenvalues of R. How large part of the total sample variance do
#    the first 3 PCs explain? The first 4? 
# 2) Estimate the matrix of factor loadings and specific variances. Calculate the communalities.
#    What do communalities show? 
# 3) Observe that the matrix of loadings gives the correlations between the input variables
#    and the factors (FA performed on R). Do any of the variables (sports events) have higher correlation
#    with factor1, factor2, factor3? Can you notice any "grouping" of the variables between the factors?
# 4) Use now VARIMAX rotation to find factor loadings that are easier to interpret. How do the loadings 
#    (correlations) change? 
# 5) Compare the communalities before and after the rotation.
#########################################################################################       
Rdec<-cor(subDec)
e      = eigen(Rdec)          # spectral decomposition of the correlation matrix
eigval = e$values[1:3]        # we consider 3 factors
eigvec = e$vectors[,1:3]
E      = matrix(eigval, nrow(Rdec), ncol = 3, byrow = T)
Q      = sqrt(E) * eigvec     # the estimated factor loadings matrix
# Alternatively:
# q1<-sqrt(eigval[1])*eigvec[,1]; # 1st column of the loadings matrix Q
# q2<-sqrt(eigval[2])*eigvec[,2]; # 2nd column of the loadings matrix Q
# q3<-sqrt(eigval[3])*eigvec[,3]; # 3rd column of the loadings matrix Q 
# Q<-cbind(q1,q2,q3)
#
com<-diag(Q%*%t(Q));          # communalities are calculated
psi <- diag(Rdec) - com       # specific variances are calculated 
tbl1<-cbind(Q,com,psi)
round(tbl1, digits=2)
sum(Q[,1]^2);                 # variance explained by factor1
sum(Q[,2]^2);                 # variance explained by factor2
sum(Q[,3]^2);                 # variance explained by factor3
sum(Q[,1]^2)+sum(Q[,2]^2)+sum(Q[,3]^2) # so much do the 3 factors together explain of the total variation
sum(com)                               # gives the same thing as the previous line
########################################
# Rotation with VARIMAX
rotQ    = varimax(Q)            # rotates the factor loadings matrix
rotLoad = rotQ$loadings         # estimated factor loadings after varimax
comrot    = diag(rotLoad %*% t(rotLoad))     # communalities with rotated loadings
psirot    = diag(Rdec) - comrot   # specific variances after rotation
tbl2    = cbind(rotLoad,comrot, psirot) # the results from FA after VARIMAX rotation, interpret!
round(tbl2, digits=2)
sum(rotLoad[,1]^2);             # variance explained by factor1
sum(rotLoad[,2]^2);             # variance explained by factor2
sum(rotLoad[,3]^2);             # variance explained by factor3
sum(rotLoad[,1]^2)+ sum(rotLoad[,2]^2)+ sum(rotLoad[,3]^2)
########################################################################################
# Perform now the analysis with k = 4 factors (describe about 78% of the total variation)
# 
eigval = e$values[1:4] # we consider 4 factors
eigvec = e$vectors[,1:4]
q1<-sqrt(eigval[1])*eigvec[,1]; # 1st column of the loadings matrix Q
q2<-sqrt(eigval[2])*eigvec[,2]; # 2nd column of the loadings matrix Q
q3<-sqrt(eigval[3])*eigvec[,3]; # 3rd column of the loadings matrix Q 
q4<-sqrt(eigval[4])*eigvec[,4]; # 4th column of the loadings matrix Q 
#
Q<-cbind(q1,q2,q3,q4);
com<-diag(Q%*%t(Q)); 
psi<-diag(Rdec)-com; 
tbl3<-cbind(Q,com,psi)
round(tbl3, digits=2)
#
rotQ    = varimax(Q)                          
rotLoad = rotQ$loadings                         
comrot  = diag(rotLoad %*% t(rotLoad))    
psirot  = diag(Rdec) - comrot  
tbl4    = cbind(rotLoad,comrot, psirot) # the results from FA after VARIMAX rotation, interpret!
round(tbl4, digits=2)
####################################













