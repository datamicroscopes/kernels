import numpy as np
from microscopes.models.mixture.dp import DPMM
from distributions.dbg.random import sample_discrete_log

def griddy_gibbs(m, hpdfs, hgrids):
    """
    run one iteration of gibbs hyperparameter estimation
    """
    # XXX: this can be done in parallel
    for fi, (hpdf, hgrid) in enumerate(zip(hpdfs, hgrids)):
        scores = np.zeros(len(hgrid))
        for i, hp in enumerate(hgrid):
            m.set_feature_hp(fi, hp)
            scores[i] = hpdf(hp) + m.score_data(fi)
        m.set_feature_hp(fi, hgrid[sample_discrete_log(scores)])

def gibbs(m, it):
    """
    run one iteration of gibbs sampling throughout the dataset
    """
    empty_gids = list(m.empty_groups())
    if not len(empty_gids):
        empty_gid = m.create_group()
    else:
        empty_gid = empty_gids[0]
        for gid in empty_gids[1:]:
            m.delete_group(gid)
    for ei, yi in it:
        gid = m.remove_entity_from_group(ei, yi)
        if not m.nentities_in_group(gid):
            m.delete_group(gid)
        idmap, scores = m.score_value(yi)
        gid = idmap[sample_discrete_log(scores)]
        m.add_entity_to_group(gid, ei, yi)
        if gid == empty_gid:
            empty_gid = m.create_group()

if __name__ == '__main__':
    from distributions.dbg.models import bb
    from microscopes.common.dataset import numpy_dataset
    import itertools as it
    import math
    def mk_bb_hyperprior_grid(n):
        return tuple({'alpha':alpha, 'beta':beta} for alpha, beta in it.product(np.linspace(0.01, 10.0, n), repeat=2))
    def bb_hyperprior_pdf(hp):
        alpha, beta = hp['alpha'], hp['beta']
        # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
        return math.pow(alpha + beta, -2.5)
    N = 10
    D = 5
    dpmm = DPMM(N, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    Y_clustered = dpmm.sample(N)
    Y = np.hstack(Y_clustered)
    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data())
    hpdfs, hgrids = (bb_hyperprior_pdf, bb_hyperprior_pdf), \
            (mk_bb_hyperprior_grid(5), mk_bb_hyperprior_grid(5))
    for _ in xrange(10):
        for _ in xrange(10):
            gibbs(dpmm, dataset.data(shuffle=True))
        griddy_gibbs(dpmm, hpdfs, hgrids)
