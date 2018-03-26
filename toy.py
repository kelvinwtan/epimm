from mvn import Mvn
import numpy as np


################
#INITIALISATION#
################


# data = {x_{0}, x_{1}, x_{2}, ... , x_{n}}

data = np.loadtxt(open("test_data/test_data1.csv", "rb"), delimiter=",")
print data

# for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var

#dimensions of the dataset
dim = data.shape[1]

#modelling each datapoint as a distribution
q_i = []

for x_i in data:
 	q_i.append(Mvn(x_i,np.identity(dim)))


#HYPERPARAMETERS
alpha = 1

#q(\theta) = \prod_i (q(\theta_i))
#where q(\theta) approximates the distribution p(D|\theta)\prod_i(p(\theta_i|\theta_i=0:n))

 


#######
# ADF #
#######

#######
#EP
#######

#Until all f_{i} converge, loop i = 1,...,n
while True:
	#Deletion

	#Incorporate new evidence


	#Update

	#If converged then break
	continue
