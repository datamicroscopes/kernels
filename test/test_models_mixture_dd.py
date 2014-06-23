from distributions.dbg.models import bb
from microscopes.models.mixture.dd import DirichletFixed
from microscopes.common.util import almost_eq
from scipy.misc import logsumexp

import numpy as np
import numpy.ma as ma
import itertools as it

from common import KL_discrete
from nose.plugins.attrib import attr

def test_sample_post_pred_no_missing_data():
    D = 5
    N = 1000
    K = 3
    alpha = 2.0

    mm = DirichletFixed(N, K, {'alpha':alpha}, [bb]*D, [{'alpha':1.,'beta':1.}]*D)

    Y_clustered, _ = mm.sample(N)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N
    mm.fill(Y_clustered)

    Y_samples = [mm.sample_post_pred(y_new=None) for _ in xrange(10000)]
    Y_samples = np.hstack(Y_samples)

    def score_post_pred(y):
        """compute log p(y | C, Y)"""
        def score_for_k(k):
            ck = mm.nentities_in_group(k)
            ctotal = mm.nentities()
            score_assign = np.log((ck + alpha/K)/(ctotal + alpha))
            score_value = sum(g.score_value(mm.get_feature_hp_shared(fi), yi) for fi, (g, yi) in enumerate(zip(mm.get_suff_stats_for_group(k), y)))
            return score_assign + score_value
        return logsumexp(np.array([score_for_k(k) for k in xrange(K)]))

    scores = np.array(list(map(score_post_pred, it.product([False, True], repeat=D))))
    scores = np.exp(scores)
    assert almost_eq(scores.sum(), 1.0)

    # lazy man
    idmap = { y : i for i, y in enumerate(it.product([False, True], repeat=D)) }

    smoothing = 1e-5
    sample_hist = np.zeros(len(idmap), dtype=np.int)
    for y in Y_samples:
        sample_hist[idmap[tuple(y)]] += 1.
    #print 'hist', sample_hist

    sample_hist = np.array(sample_hist, dtype=np.float) + smoothing
    sample_hist /= sample_hist.sum()

    #print 'actual', scores
    #print 'emp', sample_hist
    kldiv = KL_discrete(scores, sample_hist)
    print 'KL:', kldiv

    assert kldiv <= 0.05
