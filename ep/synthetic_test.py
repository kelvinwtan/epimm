import numpy as np
import ep
import time
means = np.array([[ 0.1,  0.2],
                      [ 0.8,  0.9],
                                        [-0.9,  0.5],
                                                          [-0.4,  0.1]])

covars = np.array([[ 0.005,  0.005],
                       [ 0.003,  0.003],
                                          [ 0.007,  0.007],
                                                             [ 0.009,  0.009]])

weights = np.array([ 0.4, 0.3, 0.15, 0.15])

g = ep.gmm.GMM(means=means,
        covariances=covars,
        weights=weights)
data = g.sample(50)

e = ep.epim.Epim(1.5,np.array([0,0]),np.array([[0.5,0],[0,0.5]]))
t0 = time.time()

e.fit(data)
t1 = time.time()
print("time taken:")
print(t1 -t0)