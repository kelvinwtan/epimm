import numpy as np
import sklearn.mixture
from ep.fastgmm import gmm_likelihood
import multiprocessing

class GMM():

    def __init__(self, means, covariances, weights):
        """
        Gaussian Mixture Model Distribution class for calculation of log likelihood and sampling.

        Parameters
        ----------
        means : 2-D array_like of shape (n_mixtures, n_features)
            Means for each component of the GMM
        covariances : 2-D array_like of shape (n_mixtures, n_features)
            Covariance matrices of the GMM. Only diagonal matrices are supported at this time.
        weights : 1-D array_like of shape (n_mixtures,)
            Weights for each of the GMM components
        """
        if len(covariances.shape) == 2:
            self.covariance_type = 'diag'
        else:
            raise NotImplementedError('Only diagonal covariance matrices supported')
        self.gmm = sklearn.mixture.GMM(n_components=len(weights))
        self.gmm.weights_ = weights
        self.gmm.covars_ = covariances
        self.gmm.means_ = means
        self.n_mixtures = len(weights)
        try:
            self.n_features = means.shape[1]
        except:
            raise ValueError("Means array must be 2 dimensional")

    @property
    def means(self):
        return self.gmm.means_

    @property
    def covars(self):
        return self.gmm.covars_

    @property
    def weights(self):
        return self.gmm.weights_

    @means.setter
    def means(self, means):
        # must create GMM object again so that the sklearn sample method will work correctly
        self.gmm = create_gmm(self.n_mixtures, means, self.gmm.covars_, self.gmm.weights_)

    @covars.setter
    def covars(self, covars):
        self.gmm = create_gmm(self.n_mixtures, self.gmm.means_, covars, self.gmm.weights_)

    @weights.setter
    def weights(self, weights):
        self.gmm = create_gmm(self.n_mixtures, self.gmm.means_, self.gmm.covars_, weights)

    def sample(self, n_samples):
        """
        Sample from the GMM.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
            : 2-D array_like of shape (n_samples, n_features)
            Samples drawn from the GMM distribution
        """
        return self.gmm.sample(n_samples)

    def log_likelihood(self, X, n_jobs=1):
        """
        Calculate the average log likelihood of the data given the GMM parameters

        Parameters
        ----------
        X : 2-D array_like of shape (n_samples, n_features)
            Data to be used.
        n_jobs : int
            Number of CPU cores to use in the calculation

        Returns
        -------
            : float
            average log likelihood of the data given the GMM parameters

        Notes
        -------
        For GMMs with small numbers of mixtures (<10) the use of more than 1 core can slow down the function.
        """
        n_samples = X.shape[0]
        if n_jobs == 0:
            raise ValueError("n_jobs==0 has no meaning")
        elif n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = n_jobs

        if n_jobs == 1:
            # Use the sklearn/numpy implementation
            return np.sum(self.gmm.score(X))
        else:
            # Sse compiled multireaded C code
            return gmm_likelihood(X, self.means, self.covars, self.weights, n_jobs=n_jobs)

