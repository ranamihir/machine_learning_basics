install.packages("plyr")
library("plyr")

# L1 Distance
# Assume equal weights
l1_distance <- function(c1, c2){
	cr = c("very good", "good", "medium", "poor", "very poor")
	plyr::mapvalues(cr, from = c("very good", "good", "medium", "poor", "very poor"), to = c(4, 3, 2, 1, 0))
	dist = abs(c1[1]-c2[1])/(90-15) + abs(c1[2]-c2[2])/4 + abs(c1[3]-c2[3])/4000
	return(dist)
}

# L2 Distance
l2_distance <- function(c1, c2){
	cr = c("very good", "good", "medium", "poor", "very poor")
	plyr::mapvalues(cr, from = c("very good", "good", "medium", "poor", "very poor"), to = c(4, 3, 2, 1, 0))
	dist = 0.5*(abs(c1[1]-c2[1])^2 + abs(c1[2]-c2[2])^2 + abs(c1[3]-c2[3])^2)/((c1[1]-(90+15)/2)^2 + (c2[1]-(90+15)/2)^2 + (c1[2]-(0+1+2+3+4)/5)^2 + (c2[2]-(0+1+2+3+4)/5)^2 + (c1[3]-7000)^2 + (c2[3]-7000)^2)
	return(dist)
}

# Customer 1
c1 = c(55, 3, 7000)

# Customer 2
c2 = c(25, 1, 1000)

# Calculate L1 Distance
l1_distance(c1, c2)

l2_distance(c1, c2)