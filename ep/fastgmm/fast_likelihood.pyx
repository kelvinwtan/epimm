import numpy as np
cimport numpy as np
cimport cython

cdef extern from "fast_likelihood_threaded.cpp":
    double data_logprob_threaded(double *data, double* means, double *covars, double * weights,
                        int n_samples, int n_mixtures, int n_features, int n_threads)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gmm_likelihood(np.ndarray[double, mode="c", ndim=2] X, np.ndarray[double, mode="c", ndim=2] means, np.ndarray[double, mode="c", ndim=2] covars,
                     np.ndarray[double, mode="c", ndim=1] weights, n_jobs=1):
    cdef int n_samples = X.shape[0]
    cdef int n_mixtures = weights.shape[0]
    cdef int n_features = X.shape[1]
    cdef double prob

    if n_features != means.shape[1] or n_features != covars.shape[1]:
        raise ValueError('Dimension mismatch: feature dimensionality')

    if n_mixtures != means.shape[0] or n_mixtures != covars.shape[0]:
        raise ValueError('Dimension mismatch: number of mixtures')

    prob = data_logprob_threaded(<double*> X.data, <double*> means.data, <double*> covars.data, <double*> weights.data,
                n_samples, n_mixtures, n_features, n_jobs)

    return prob
