from microscopes.models.mixture.dp import DPMM
from distributions.dbg.models import bb

import numpy as np

def bb_hyperprior_pdf(hp):
    alpha, beta = hp['alpha'], hp['beta']
    # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
    return -2.5 * np.log(alpha + beta)

def make_one_feature_bb_mm(Nk, K, alpha, beta):
    dpmm = DPMM(K*Nk, {'alpha':2.0}, [bb], [{'alpha':alpha,'beta':beta}])
    shared = bb.Shared()
    shared.load({'alpha':alpha,'beta':beta})
    def init_sampler():
        samp = bb.Sampler()
        samp.init(shared)
        return samp
    samplers = [init_sampler() for _ in xrange(K)]
    def gen_cluster(samp):
        data = [(samp.eval(shared),) for _ in xrange(Nk)]
        return np.array(data, dtype=[('', bool)])
    Y_clustered = tuple(map(gen_cluster, samplers))
    dpmm.fill(Y_clustered)
    return dpmm
