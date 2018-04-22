import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal as mv
"""
object class to represent a multivariate normal distribution
as well as functions that allow for multiplying and dividing mvns together
"""

class Mvn:

	"""
	Each Mvn is described by a d-sized vector mean and a dd-sized covariance matrix
	"""
	def __init__(self, mu, sig,s=1):
		self.mu = mu
		self.sig = sig
                self.s = s
	
	def __mul__(self, other):
		sig_hat = inv(inv(self.sig) + inv(other.sig))
		mu_hat  = sig_hat.dot((inv(self.sig).dot(self.mu) + inv(other.sig).dot(other.mu)))
                return Mvn(mu_hat, sig_hat, s=self.s*other.s)

	def __div__(self, other):
		sig_hat = inv(inv(self.sig) - inv(other.sig))
		mu_hat  = sig_hat.dot((inv(self.sig).dot(self.mu) - inv(other.sig).dot(other.mu)))
		return Mvn(mu_hat, sig_hat, s=self.s/other.s)

	def pdf(self, d):
	    """
            Returns PDF of MVN at point d
            """
            return mv.pdf(d, mean=self.mu, cov=self.sig)
   
        def logpdf(self, d):
            return mv.logpdf(d, mean=self.mu, cov=self.sig)
