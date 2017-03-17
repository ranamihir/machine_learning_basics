### Exercise 1 ###

# Load data
uscrime = readRDS("uscrime.rds")

# Describe data
str(uscrime)
summary(uscrime)
head(uscrime)

regs = factor(uscrime$reg)
uscrime$reg = as.numeric(regs)

# Make copy with only required columns
data = uscrime[, c("murder", "rape", "robbery", "assault", "burglary", "larceny", "autotheft", "reg")]


## Part 1) ##

# Mean normalized data
scaled_data = scale(data, center=TRUE, scale=TRUE)

# Compare the covariance matrices of the generated data and centered data
cov_data = cov(data)
cov_scaled_data = cov(scaled_data)

# Compare the means before and after normalization
data_mean = colMeans(data)
scaled_data_mean = colMeans(scaled_data)


## Part 2) ##

corr = cov(scaled_data)

# Find the spectral decomposition of corr
eig = eigen(corr)

# Find eigen vectors of corr
V = eig$vectors
V

# Linear transformation with the matrix of eigenvectors
data_transformed =  scaled_data %*% V

# Find proportion of variance explained by first 2 PCs
sum(var(data_transformed[, 1:2]))/sum(var(data_transformed))
# Find proportion of variance explained by first 3 PCs
sum(var(data_transformed[, 1:3]))/sum(var(data_transformed))

# Compare with the results from PCA
# Calculate individual and cumulative proportion of variance explained
prin_comp = prcomp(data, center=TRUE, scale.=TRUE)
pr_var = prin_comp$sdev^2
ind_prop_var_exp = pr_var/sum(pr_var)
ind_prop_var_exp

# Calculate individual proportion of variance explained
plot(ind_prop_var_exp, xlab="Principal Component", ylab="Proportion of Variance Explained", type="b")

# Calculate cumulative proportion of variance explained
plot(cumsum(ind_prop_var_exp), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", type="b")

# Find proportion of variance explained by first 2 PCs
sum(pr_var[1:2])/sum(pr_var)
# Find proportion of variance explained by first 3 PCs
sum(pr_var[1:3])/sum(pr_var)

# Direct method of finding all the proportions
summary(prin_comp)

# Formula for first 2 PCs
pc1 = scaled_data %*% prin_comp$rotation[, 1]
pc1
pc2 = scaled_data %*% prin_comp$rotation[, 2]
pc2


## Part 3) ##

# Correlation between variables and PCs
var_cor_func <- function(var.loadings, comp.sdev) {
  var.loadings*comp.sdev
}

# Variable correlation / coordinates
loadings = prin_comp$rotation
sdev = prin_comp$sdev
var.cor = t(apply(loadings, 1, var_cor_func, sdev))

# Correlation between variables and first 2 PCs
var.cor[, 1:2]

# Plot the correlation circle
a = seq(0, 2*pi, length=100)
plot(cos(a), sin(a), type='l', col="gray", xlab="PC1",  ylab="PC2")
abline(h=0, v=0, lty=2)
# Add variables
arrows(0, 0, var.cor[, 1], var.cor[, 2], length=0.1, angle=15, code=2)
# Add labels
text(var.cor, labels=rownames(var.cor), cex=1, adj=1)

## Part 4) ##

# PC scores (new coordinates) for the 10 first data points
prin_comp$x[1:10,]

# Scatterplot of first 2 PC scores labelled by region
plot(prin_comp$x[, 1], prin_comp$x[, 2], main="Scatterplot of First 2 PC scores", xlab="PC1", ylab="PC2", pch=19)
text(prin_comp$x[, 1], prin_comp$x[, 2], labels=uscrime$reg, cex=0.7, adj=2.5)