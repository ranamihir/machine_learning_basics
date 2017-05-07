### Exercise 2 ###

# install.packages("psych")
# library(psych)

# Load data
uscrime = readRDS("uscrime.rds")

# Describe data
str(uscrime)
summary(uscrime)
head(uscrime)

# Make copy with only required columns
data = uscrime[, c("murder", "rape", "robbery", "assault", "burglary", "larceny", "autotheft")]

# Check degrees of freedom
p = 7
m = seq(1, 5, by=1)
m
dof = ((p-m)^2 - (p+m))/2
dof


# Mean normalized data
scaled_data = scale(data, center=TRUE, scale=TRUE)

## Part 2.1 ##
# FA using Principal Components

corr = cor(data)
e = eigen(corr)

# k = 2 factors
eigval = e$values[1:2]
eigvec = e$vectors[, 1:2]

# Loadings
q1 = sqrt(eigval[1]) * eigvec[, 1]
q2 = sqrt(eigval[2]) * eigvec[, 2]
Q = cbind(q1, q2)

# Total estimate of communalities
com = diag(Q %*% t(Q))
sum(com)

# Or
psi  =  diag(corr) - com
tbl1 = cbind(Q, com, psi)
round(tbl1, digits=4)
sum(Q[, 1]^2)
sum(Q[, 2]^2)
sum(Q[, 1]^2) + sum(Q[, 2]^2)

# Rotation with VARIMAX
rotQ = varimax(Q)
rotLoad = rotQ$loadings
comrot = diag(rotLoad %*% t(rotLoad))
psirot = diag(corr) - comrot
tbl2 = cbind(rotLoad, comrot, psirot)
round(tbl2, digits=4)
sum(rotLoad[, 1]^2)
sum(rotLoad[, 2]^2)
sum(rotLoad[, 1]^2) + sum(rotLoad[, 2]^2)


# k = 3 factors
eigval = e$values[1:3]
eigvec = e$vectors[, 1:3]

# Loadings
q1 = sqrt(eigval[1]) * eigvec[, 1]
q2 = sqrt(eigval[2]) * eigvec[, 2]
q3 = sqrt(eigval[3]) * eigvec[, 3]
Q = cbind(q1, q2, q3)

# Total estimate of communalities
com = diag(Q %*% t(Q))
sum(com)

# Or
psi  =  diag(corr) - com
tbl1 = cbind(Q, com, psi)
round(tbl1, digits=4)
sum(Q[, 1]^2)
sum(Q[, 2]^2)
sum(Q[, 3]^2)
sum(Q[, 1]^2) + sum(Q[, 2]^2) + sum(Q[, 3]^2)


# Rotation with VARIMAX
rotQ = varimax(Q)
rotLoad = rotQ$loadings
comrot = diag(rotLoad %*% t(rotLoad))
psirot = diag(corr) - comrot
tbl2 = cbind(rotLoad, comrot, psirot)
round(tbl2, digits=4)
sum(rotLoad[, 1]^2)
sum(rotLoad[, 2]^2)
sum(rotLoad[, 3]^2)
sum(rotLoad[, 1]^2) + sum(rotLoad[, 2]^2) + sum(rotLoad[, 3]^2)

resid = corr - rotLoad %*% t(rotLoad)
sqrt(sum((resid - diag(psirot))^2)/(7*6))


## Part 2.2 ##
# FA using Principal Factor Method

# k = 2 factors
pfa = fa(scaled_data, nfactors=2, fm="pa", SMC=TRUE, rotate="varimax", max.iter=50)
print(pfa)
pfatable = cbind(pfa$loadings, (1 - pfa$uniquenesses), pfa$uniquenesses)
colnames(pfatable) = c("q1", "q2", "Communalities", "Specific variances")
print(round(pfatable,2))

pfa$rms
resid = corr - pfa$loadings %*% t(pfa$loadings)
sqrt(sum((resid - diag(pfa$uniquenesses))^2) / (7*6))

## Part 4 ##
plot(pfa$scores[, 1], pfa$scores[, 2])
text(pfa$scores[, 1], pfa$scores[, 2], labels=uscrime$reg, cex=0.7, adj=2.5)
rownames(pfa$scores)[which.max(pfa$scores[,1])]
rownames(pfa$scores)[which.max(pfa$scores[,2])]

# k = 3 factors
pfa = fa(scaled_data, nfactors=3, fm="pa", SMC=TRUE, rotate="varimax", max.iter=50)
print(pfa)
pfatable = cbind(pfa$loadings, (1 - pfa$uniquenesses), pfa$uniquenesses)
colnames(pfatable) = c("q1", "q2", "q3", "Communalities", "Specific variances")
print(round(pfatable,2))

pfa$rms
resid = corr - pfa$loadings %*% t(pfa$loadings)
sqrt(sum((resid - diag(pfa$uniquenesses))^2) / (7*6))

## Part 4 ##
plot(pfa$scores[, 1], pfa$scores[, 3])
text(pfa$scores[, 1], pfa$scores[, 2], labels=uscrime$reg, cex=0.7, adj=2.5)
rownames(pfa$scores)[which.max(pfa$scores[,1])]
rownames(pfa$scores)[which.max(pfa$scores[,2])]
rownames(pfa$scores)[which.max(pfa$scores[,3])]


## Part 2.3 ##
# FA using Maximum Likelihood Method

# k = 2 factors
mlm  = factanal(scaled_data, factors=2, rotation="varimax")
print(mlm)

# Or
mlmtable = cbind(mlm$loadings, (1 - mlm$uniquenesses), mlm$uniquenesses)
colnames(mlmtable) = c("q1", "q2", "Communalities", "Specific variances")
print(round(mlmtable,2))


# k = 3 factors
mlm  = factanal(scaled_data, factors=3, rotation="varimax")
print(mlm)

# Or
mlmtable = cbind(mlm$loadings, (1 - mlm$uniquenesses), mlm$uniquenesses)
colnames(mlmtable) = c("q1", "q2", "q3", "Communalities", "Specific variances")
print(round(mlmtable,2))