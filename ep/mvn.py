import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal as mv
"""
object class to represent a multivariate normal distribution
a wrapper around the scipy.stats multivariate normal
as well as functions that allow for multiplying and dividing mvns together
"""

class Mvn:

    """
    Each Mvn is described by a d-sized vector mean and a dd-sized covariance matrix
    """
    def __init__(self, mean, precision, weight = 1):
        #self.mvn = mv(mean, inv(precision))
        self.mean = mean
        self.precision = precision
        self.weight = weight

    def __mul__(self, other):
        if type(other) is int:
            return self
        precision_new = self.precision + other.precision
        mean_new  =  np.dot(inv(precision_new), np.dot(self.precision,self.mean) + np.dot(other.precision, other.mean))
        
        p = Mvn(other.mean, inv(inv(other.precision) + inv(self.precision)))
        
        r = p.pdf(self.mean) * self.weight * other.weight
        if r < 1e-200:
            weight_new = 1e-200
        else:
            weight_new = r

        return Mvn(mean = mean_new, precision = precision_new, weight = weight_new)
    
    def __rmul__(self, other):
        if type(other) is int:
            return self
        precision_new = self.precision + other.precision
        mean_new  =  np.dot(inv(precision_new), np.dot(self.precision,self.mean) + np.dot(other.precision, other.mean))
        
        p = Mvn(other.mean, inv(inv(other.precision) + inv(self.precision)))


        r = p.pdf(self.mean) * self.weight * other.weight
        if r < 1e-200:
            weight_new = 1e-200
        else:
            weight_new = r

        return Mvn(mean = mean_new, precision = precision_new, weight = weight_new)

    def __div__(self, other):
        if type(other) is int:
            return self
        precision_new = self.precision - other.precision
        mean_new  =  np.dot(inv(precision_new), np.dot(self.precision,self.mean) - np.dot(other.precision, other.mean))
        

        a = np.linalg.det(other.precision)
        b = np.linalg.det(inv(other.precision) - inv(self.precision))

        w = a/b
        p = Mvn(other.mean, inv(inv(other.precision) - inv(self.precision)))
        weight_new = (w/p.pdf(self.mean))*self.weight/other.weight

        return Mvn(mean = mean_new, precision = precision_new, weight = weight_new)

    def __rdiv__(self, other):
        raise NotImplementedError
        print("Use inv matrix to do division for now")

        @property
        def mean(self):
            return self.mean

        @mean.setter
        def mean(self, m):
            self.mean = m 

        @property
        def pres(self):
            return self.precision

        @precision.setter
        def pres(self, v):
            self.precision = v

        @property
        def pres(self):
            return self.weight

        @weight.setter
        def pres(self, v):
            self.weight = v

    def pdf(self, x):

        """
        Returns PDF of MVN at point x
        """
        m = np.atleast_2d(x - self.mean)
        expo = -0.5*np.dot(np.dot(m,self.precision),m.T)
        s = (2*np.pi)**m.shape[1]

        #test =  mv.pdf(x, mean=self.mean, cov=inv(self.precision))
        ret = np.sqrt(np.linalg.det(self.precision)/s) * np.exp(expo)
        # print("nan???",self.mean, self.precision, x,ret[0][0]*self.weight)
        # print(self.weight, self.precision)
        if ret[0][0] * self.weight < 1e-200:
            return 1e-200

        return ret[0][0] * self.weight
        #return mv.pdf(x, mean=self.mean, cov=inv(self.precision))

    def logpdf(self, x):
        return mv.logpdf(x, mean=self.mean, cov=inv(self.precision))
