import numpy as np
import ep
import time
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

means = np.array([[ 2,  2] , [-1 , -1]])

covars = np.array([[ 0.5 ,0.5], [0.5, 0.5]])

weights = np.array([0.5,0.5])




# means = np.array([[ 2,  5]])

# covars = np.array([[ 0.1,  0.1]])

# weights = np.array([1])

g = ep.gmm.GMM(means=means,
        covariances=covars,
        weights=weights)
data = g.sample(100)

e = ep.epim.Epim(0.5,np.array([0,0]),np.linalg.inv(np.array([[2500,0],[0,2500]])))
t0 = time.time()

e.fit(data)
t1 = time.time()
print("time taken:")
print(t1 -t0)
