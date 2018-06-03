import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal as mv


x = np.array([0.50420473, 0.50092741])
m = np.array([0.50440124, 0.50112264])
v = np.array([[0.00270442, 0.00268684],[0.00268684, 0.00266938]])

covar = [[2.8973310178240777e-5, 4.2758602306086014e-5],[4.2758602306086014e-5, 6.3102837056674781e-5]]
a = inv(covar)
print (a)

a = mv(m,v)
r = a.pdf(x)
print(r)