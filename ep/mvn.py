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
    def __init__(self, mean, covars):
        self.mvn = mv(mean, covars)
        self.mean = mean
        self.covar = covars
                
    def __mul__(self, other):
        if isinstance(other, int):
            return self
        covar_hat = inv(inv(self.covar) + inv(other.covar))
        mean_hat  =  np.dot(covar_hat, np.dot(inv(self.covar),self.mean) + np.dot(inv(other.covar), other.mean))
        return Mvn(mean_hat, covar_hat)

    def __div__(self, other):
        if isinstance(other, int):
            return self
        covar_hat = inv(inv(self.covar) - inv(other.covar))
        mean_hat  =  np.dot(covar_hat, np.dot(inv(self.covar),self.mean) - np.dot(inv(other.covar), other.mean))
        return Mvn(mean_hat, covar_hat)

        @property
        def mean(self):
            return self.mean

        @mean.setter
        def mean(self, m):
            self.mean = m 

        @property
        def covar(self):
            return self.covar

        @covar.setter
        def covar(self, v):
            self.covar = v

    def pdf(self, x):
        """
        Returns PDF of MVN at point x
        """
        return self.mvn.pdf(x)
   
    def logpdf(self, x):
        return self.mvn.logpdf(x)
