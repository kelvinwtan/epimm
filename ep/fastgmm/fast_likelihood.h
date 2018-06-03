#include <stdio.h>

double data_logprob_threaded(double * __restrict data,
                             double * __restrict means,
                             double * __restrict covars,
                             double * __restrict weights,
                             int n_samples,
                             int n_mixtures,
                             int n_features,
                             int n_threads);