from microscopes.kernels.gibbs import gibbs_hp
from common import make_one_feature_bb_mm, bb_hyperprior_pdf

import itertools as it
import numpy as np
import scipy as sp
import scipy.misc

from nose.plugins.attrib import attr

@attr('slow')
def test_kernel_gibbs_hp():
    def mk_bb_hyperprior_grid(n):
        return tuple({'alpha':alpha,'beta':beta} for alpha, beta in it.product(np.linspace(0.01, 5.0, n), repeat=2))
    Nk = 1000
    K = 100
    gridsize = 10
    dpmm = make_one_feature_bb_mm(Nk, K, 1.0, 1.0)
    dpmm.set_feature_hp(0, {'alpha':1.5,'beta':3.0}) # don't start w/ the right answer

    # find closest grid point to actual point
    grid = mk_bb_hyperprior_grid(gridsize)
    dists = np.array([np.linalg.norm(np.array([g['alpha'], g['beta']]) - np.array([1., 1.])) for g in grid])
    closest = grid[np.argmin(dists)]
    closest = np.array([closest['alpha'], closest['beta']])

    hpdfs, hgrids = (bb_hyperprior_pdf,), (grid,)
    def posterior(k, skip):
        for _ in xrange(k):
            for _ in xrange(skip-1):
                gibbs_hp(dpmm, hpdfs, hgrids)
            gibbs_hp(dpmm, hpdfs, hgrids)
            hp = dpmm.get_feature_hp(0)
            yield np.array([hp['alpha'], hp['beta']])
    values = list(posterior(100, 10))
    avg = sum(values) / len(values)

    #print closest, avg
    #print np.linalg.norm( closest - avg )
    assert np.linalg.norm( closest - avg ) <= 0.1

    #import matplotlib.pylab as plt
    #plt.plot( [1.], [1.], 'ro' )
    #plt.scatter( [v[0] for v in values] + np.random.random(size=len(values))*0.1,
    #             [v[1] for v in values] + np.random.random(size=len(values))*0.1,
    #             alpha=0.2 )
    #plt.xlim(xmin=0., xmax=5.0)
    #plt.ylim(ymin=0., ymax=5.0)
    #plt.savefig('foo.pdf')
