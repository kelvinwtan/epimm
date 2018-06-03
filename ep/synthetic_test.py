import numpy as np
import ep
import time
import pickle
from numpy.linalg import inv
"""
means = np.array([[ 0.1,  0.2],
                      [ 0.8,  0.9],
                                        [-0.9,  0.5],
                                                          [-0.4,  0.1]])

covars = np.array([[ 0.005,  0.005],
                       [ 0.003,  0.003],
                                          [ 0.007,  0.007],
                                                             [ 0.009,  0.009]])

weights = np.array([ 0.4, 0.3, 0.15, 0.15])
"""

means = np.array([[1,  1] , [-1 , -1]])

covars = np.array([[ 0.5 ,0.5], [0.5, 0.5]])

weights = np.array([0.5,0.5])




# means = np.array([[ 2,  5]])

# covars = np.array([[ 0.1,  0.1]])

# weights = np.array([1])

g = ep.gmm.GMM(means=means,
        covariances=covars,
        weights=weights)
data = g.sample(100)


base_mean = np.array([0,0])
base_covar = np.linalg.inv(np.array([[2500,0],[0,2500]]))
set_covar = inv(np.eye(len(base_mean))*10000)
alpha = 0.5

a= ep.adf.ADFIMM(alpha,base_mean,base_covar,set_covar)
t0 = time.time()

q_i, t_ij = a.factorise_prior(data)


b = ep.epim.EPIMM(alpha,base_mean,base_covar,set_covar)
b.fit(data,q_i,t_ij)

# with open('adf_qi.pickle','wb') as f:
# 	pickle.dump(q_i, f)
# with open('adf_tij.pickle','wb') as f2:
# 	pickle.dump(t_ij, f2)

t1 = time.time()
print("time taken:")
print(t1 -t0)
