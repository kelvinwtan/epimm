from mvn import Mvn
import numpy as np

a = Mvn([1,2,3,4],np.eye(4))
b = Mvn([-1,0,0,-1], [[5, 0, 0, 0],[0, 2, 0, 0],[0, 0, 2, 0],[ 0, 0, 0, 5]])

c = a*b
d = a/b


likelihood = c.logpdf([0.333,1.3333,2,3.1667])
print("likelihood = %s" % likelihood)
