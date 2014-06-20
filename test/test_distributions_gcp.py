from microscopes.common.util import random_orthonormal_matrix
from microscopes.distributions.gcp import sample_niw, sample_iw

import numpy as np

#import rpy2.robjects as ro
#from rpy2.robjects.numpy2ri import activate
#activate()
#ro.r('library(MCMCpack)')

def test_iw_sampler():
    Q = random_orthonormal_matrix(2)
    nu = 4
    S = np.dot(Q, np.dot(np.diag([1.0, 0.5]), Q.T))
    invS = np.linalg.inv(S)
    #def r_sample_iw(nu, scale):
    #    return ro.r.riwish(nu, scale)
    #r_samples = [r_sample_iw(nu, S) for _ in xrange(10000)]
    py_samples = [sample_iw(nu, S) for _ in xrange(10000)]

    true_mean = 1./(nu-S.shape[0]-1)*S
    #r_mean = sum(r_samples) / len(r_samples)
    py_mean = sum(py_samples) / len(py_samples)

    print 'true:', true_mean
    #print 'r:', r_mean
    print 'py:', py_mean
    diff = np.linalg.norm(true_mean - py_mean)
    print 'F-norm:', diff
    assert diff <= 0.3
