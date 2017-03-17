# Visualization example with Swiss bank data.  
#
bank<-read.table("SwissBankNotes.txt",header=T)
fix(bank)
group<-c(rep(1,100),rep(2,100))
bank$group<-group
attach(bank) # variables Length, HeightL, HeightR, InnerL, InnerU, Diag
#
boxplot(Diag ~ group); title("Diagonal of bank notes")
boxplot(Length~group); title("Length variable of bank notes")
#
# Alternative parallel boxplot with mean bars
x  = bank
m1 = mean(x[1:100, 6])
m2 = mean(x[101:200, 6])
# plot
boxplot(x[1:100, 6], x[101:200, 6], axes = FALSE, frame = TRUE)
axis(side = 1, at = seq(1, 2), label = c("GENUINE", "COUNTERFEIT"))
axis(side = 2, at = seq(130, 150, 0.5), label = seq(130, 150, 0.5))
title("Swiss Bank Notes")
lines(c(0.6, 1.4), c(m1, m1), lty = "dotted", lwd = 1.2)
lines(c(1.6, 2.4), c(m2, m2), lty = "dotted", lwd = 1.2)
# boxplot(HeightL~group); boxplot(HeightR~group); 
# boxplot(InnerL~group); boxplot(InnerU~group); 
# 
# Scatter plots
#
plot(InnerU,Diag,pch=group,col = group) # plotting character and colour given by group
#install.packages ("scatterplot3d")
library(scatterplot3d)
scatterplot3d(InnerL,InnerU,Diag,pch=group,color=group) # plotting character and colour given by group
#
# Study the variables pairwise, at the same time
plot(bank[,3:6],pch=group,col = group); title("Pairwise scatter plots")
round(cor(bank[,-7]), 2)
#
#############################################################################################
# Ex.1. Car data (Chambers, Cleveland, Kleiner & Tukey, 1983). Consists of 13 variables measured for 
# 74 car types. The variables: P = price; M = mileage (miles per gallon); 
# R78 = repair record 1978 (rated on a 5-point scale; 5 best, 1 worst);
# R77 = repair record 1977 (scale as before), H = headrooom (in inches); 
# R = rear seat clearance(distance from front seat back to rear seat, in inches);
# Tr = trunk space (in cubic feet); W = weight (in pound); L = length (in inches);
# T = turning diameter (clearance required to make a U-turn, in feet); 
# D = displacement (in cubic inches); G = gear ratio for high gear;
# C = company headquarter 
#
# 1)
carc<-readRDS("cardata.rds")
x<-data.frame(carc$M,carc$C); x1<-subset(x,x[,2]=="US"); x2<-subset(x,x[,2]=="Japan"); x3<-subset(x,x[,2]=="Europe");
summary(x1$carc.M)
boxplot(carc$M~carc$C); 
#
par(mfrow=c(2,2)) 
hist(carc$M[carc$C=="US"],main="U.S. cars",xlim=c(10,45),xlab="")
hist(carc$M[carc$C=="Japan"],main="Japanese cars",xlim=c(10,45),xlab="")
hist(carc$M[carc$C=="Europe"],main="European cars",xlim=c(10,45),xlab="")
hist(carc$M,main="All cars",xlim=c(10,45),xlab="")
#
grvar<-as.numeric(carc$C); 
class(grvar)
plot(carc$M,carc$W, pch = grvar, col = grvar, xlab = "Mileage (X2)", ylab = "Weight (X8)", main = "Car Data")
#############################################################################################
# Ex.3
A<-matrix(c(1,2,-1,4),nrow=2,byrow=T)
y<-eigen(A); 
y$vec; y$val
#help(eigen)
#
# Ex.4
B<-matrix(c(3,1,1,1,0,2,1,2,0),nrow=3,byrow=T)
y1<-eigen(B); 
V<-y1$vec; L<-diag(y1$val)
t(V)%*%B%*%V
V%*%L%*%t(V)
#
# Ex.5
C<-matrix(c(3,6,-1,6,9,4,-1,4,3),nrow=3,byrow=T)
y2<-eigen(C); 
V<-y2$vec; L<-diag(y2$val)
#
C2<-C%*%C
y3<-eigen(C2); y3$vec; y3$val
#
C3<-solve(C); C3%*%C 
y4<-eigen(C3); y4$vec; y4$val
1/diag(L) 
############################################################################################
# Ex.6 (SVD)
A<-matrix(c(4,-5,-1,7,-2,3,-1,4,-3,8,2,6),nrow=4,byrow=T)
A
y<-svd(A)
D<-diag(y$d); U<-y$u; V<-y$v
D; U; V;
A1<-A%*%t(A); A2<-t(A)%*%A
y1<-eigen(A1); y2<-eigen(A2)
y1$vec; y1$val; y2$vec; y2$val