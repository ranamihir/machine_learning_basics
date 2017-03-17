#############################################################################################
# Principal component analysis
#############################################################################################
# Example 1. Linear transformations example. The purpose is to demonstrate how PCA works. 
# 
# Generate data from a 2-dim normal distribution. 
 library(mvtnorm)
#
n   = 200 
rho = 0.8; mu  = c(2, 2); Sig = matrix(c(1, rho, rho, 1), nrow = 2)
x  = rmvnorm(n, mu, Sig)
xc <- scale(x, center = TRUE, scale=FALSE) # center the data
cov(x); cov(xc)  # Compare the covariance matrices of the generated data and centered data
colMeans(x); colMeans(xc);
par(mfrow=c(1,2))
plot(x[,1],x[,2]); plot(xc[,1],xc[,2])
#   
sig<-cov(xc)
eig = eigen(sig) # Find the spectral decomposition of 'sig'
L <-diag(eig$values); V <- eig$vectors  # V%*%L%*%t(V)
L; V
# linear transformation with the matrix of eigenvectors
xtr<- xc %*% V    
var(xtr[,1]); var(xtr[,2]) # variances of the new variables
cor(xtr[,1],xtr[,2]) # correlation between the new variables
# Scatter plots for the generated (centered) and transformed data 
plot(xc[,1],xc[,2],xlim=c(-3.2,3.2),ylim=c(-3.2,3.2)); 
plot (xtr[,1],xtr[,2], xlim=c(-4,4), ylim=c(-4,4))
#
# Compare with the results from PCA
pcaxc<-prcomp(x)
summary(pcaxc); 
pcaxc$sdev^2
print(pcaxc)
dim(pcaxc$x); 
pcaxc$x[1:10,]  # PC scores (new coordinates) for the 10 first data points
#
dim(xtr); xtr[1:10,] # ten first transformed data points 
#
# Arbitrary linear transformation (rotation)
dir1 = c(-1,0.5); dir2 = c(-0.5,-1) # choose a direction dir1 and then dir2 which is orthogonal to dir1
norm = c(t(dir1) %*% dir1, t(dir2) %*% dir2) # normalize the vectors 
dir1 = dir1 / sqrt(norm[1])
dir2 = dir2 / sqrt(norm[2])
a<-matrix(c(dir1,dir2),nrow=2,ncol=2,byrow=F)
# 
axtr<-xc%*%a  # linear transformation
var(axtr[,1]); var(axtr[,2]) # variance of the new variables, check also with t(a)%*%sig%*%a
cor(axtr[,1],axtr[,2]) # correlation between the new variables 
plot(axtr[,1],axtr[,2]) 
############################################################################################
# Example 2. Olympic decathlon data from 1988. 
# Perform PCA on the decathlon data. We start with removing one observation which is very 
# different from the rest. Think also about the scales of the variables, so that 'large' 
# values correspond to a 'better' result. 
# 
decathlon<-read.table("olympicDec.txt",header=T)
fix(decathlon)
dim(decathlon)
boxplot(decathlon[,12])
# Remove the last object and form a data set with only 10 sports events
subDec<-decathlon[-34,2:11]; fix(subDec)
# New variables for the running events
subDec$run100<--subDec$run100; subDec$run400<--subDec$run400; subDec$hurdle<--subDec$hurdle;
subDec$run1500<--subDec$run1500;
colMeans(subDec); apply(subDec, 2, sd) # 2 in apply indicates that 'sd' will be applied to columns
# Study correlations
pairs(subDec[,1:5]); pairs(subDec[,5:10]); round(cor(subDec),2)
# Perform now PCA. Since the scales of the variables are so different, we use the correlation matrix.
decpca <- prcomp(subDec, scale = TRUE); 
names(decpca) # Study the components of 'decpca', what is what
print(decpca); summary(decpca)
# The variances of the principal components
decpca$sdev^2
# Study the covariances and correlations of the PCs
# The matrix decpca$x contains the PC scores, our new variables
round(cov(decpca$x), 8); round(cor(decpca$x), 2)
#
# Study the correlation between the "Score" and the first 2 PCs.
par(mfrow=c(1,2))
plot(decpca$x[,1], decathlon$score[-34])
plot(decpca$x[,2], decathlon$score[-34])
cor(decpca$x[,1], decathlon$score[-34]); cor(decpca$x[,2], decathlon$score[-34])
#
rank(decathlon$score[-34]); rank(decpca$x[,1]) # Rankings: according to 'score' and according to the first PC
screeplot(decpca, type="lines")
#
biplot(decpca, scale=0, col = c("blue", "black"))
#
############################################################################################
# Example 3. Swiss bank notes example.
#
bank<-read.table("SwissBankNotes.txt",header=T)
fix(bank)
# Recall the explanatory analysis from the last time
bank$group<-c(rep(1,100),rep(2,100))
attach(bank) # variables Length, HeightL, HeightR, InnerL, InnerU, Diag
boxplot(Length~group); boxplot(HeightL~group); boxplot(HeightR~group); 
boxplot(InnerL~group); boxplot(InnerU~group); boxplot(Diag ~ group); 
# Study the variables pairwise
plot(bank[,-7],pch=group,col=group)
round(cor(bank[,-7]), 2)
#
plot(InnerU,Diag,pch=group,col=group) # plotting character given by group
scatterplot3d(InnerL,InnerU,Diag,pch=group,color=group) # scatterplot3D, plotting character given by group
#
colMeans(bank[,-7]); apply(bank[,-7], 2, sd)
#######################################################
# PCA, we use the covariance matrix, that is we do not 
# standardize the data.
#
bankpca <- prcomp(bank[,-7], scale = FALSE)
print(bankpca); names(bankpca)
bankpca$sdev^2
summary(bankpca)
#
par(mfrow=c(2,2))
plot(bankpca$x[,1], bankpca$x[,2],pch=group, col=group)
plot(bankpca$x[,2], bankpca$x[,3],pch=group, col=group)
plot(bankpca$x[,1], bankpca$x[,3],pch=group, col=group)
screeplot(bankpca)
#
biplot(bankpca, choices=1:2, scale=0, col = c("blue", "black"))
#biplot(bankpca, choices=c(1,3), scale=0, col = c("blue", "black"))
#
# The first two PCs 'manually'.
class(bank[,-7])
hm <- as.matrix(bank[,-7])
newd<-scale(hm, center = TRUE, scale = FALSE)
pcvec1<-newd%*%bankpca$rotation[,1]
pcvec2<-newd%*%bankpca$rotation[,2]
plot(pcvec1,pcvec2,pch=group, col=group)
#######################################################
# Now we use the correlation matrix
#
bankpca2 <- prcomp(bank[,-7], scale = TRUE)
print(bankpca2); names(bankpca2)
bankpca2$sdev^2
summary(bankpca2)
#
par(mfrow=c(2,2))
plot(bankpca2$x[,1], bankpca2$x[,2],pch=group, col=group)
plot(bankpca2$x[,2], bankpca2$x[,3],pch=group, col=group)
plot(bankpca2$x[,1], bankpca2$x[,3],pch=group, col=group)
screeplot(bankpca2)
#
biplot(bankpca2, choices=1:2, scale=0, col = c("blue", "black"))
#
##########################################################
# Circle with correlations for the first two PCs.
#
r    = cor(cbind(bankpca$x,bank[,-7]))  # correlations between PCs and original variables
r1   = r[7:12, 1:2]                     # Consider the correlations between the original variables and PC1, PC2
r2   = r[7:12, 1:6]  
sum(r2[1,]^2)  
#
#dev.new()
ucircle = cbind(cos((0:360)/180 * pi), sin((0:360)/180 * pi))
plot(ucircle, type = "l", lty = "solid", col = "blue", xlab = "First PC", ylab = "Second PC", 
    main = "Swiss Bank Notes", cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8, lwd = 2)
abline(h = 0, v = 0)
label = c("X1", "X2", "X3", "X4", "X5", "X6")
text(r1, label, cex = 1.2) 
##############################################################################################



