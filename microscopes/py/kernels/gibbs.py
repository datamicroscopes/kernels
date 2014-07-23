import numpy as np
from distributions.dbg.random import sample_discrete_log

###
# NOTE: while assign_fixed(), assign(), and assign_resample()
# are all very similar and could be merged into one function with various input
# parameters governing the behavior, such an approach is confusing and error
# prone!
#
# Since the underlying gibbs kernel is quite concise, we trade off slight code
# duplication for increased clarity, less bugs, and slightly better performance
# (since extra runtime checks add up)
###

def assign_fixed(s, rng=None):
    for ei in np.random.permutation(s.nentities()):
        s.remove_value(ei, rng)
        idmap, scores = s.score_value(ei, rng)
        gid = idmap[sample_discrete_log(scores)]
        s.add_value(gid, ei, rng)

def assign(s, rng=None):
    empty_gids = list(s.empty_groups())
    if not len(empty_gids):
        empty_gid = s.create_group(rng)
    else:
        empty_gid = empty_gids[0]
        for gid in empty_gids[1:]:
            s.delete_group(gid)
    assert len(s.empty_groups()) == 1
    for ei in np.random.permutation(s.nentities()):
        gid = s.remove_value(ei, rng)
        if not s.groupsize(gid):
            # we have already ensured exactly one empty group (before removing
            # the current entity).  now that we have two empty groups, we must
            # remove one of them
            s.delete_group(gid)
        idmap, scores = s.score_value(ei, rng)
        gid = idmap[sample_discrete_log(scores)]
        s.add_value(gid, ei, rng)
        if gid == empty_gid:
            # we used up our one empty group, so we must create another
            empty_gid = s.create_group(rng)

def assign_resample(s, nonempty, rng=None):
    if nonempty <= 0:
        raise ValueError("nonempty needs to be >= 1")
    for ei in np.random.permutation(s.nentities()):
        gid = s.remove_value(ei, rng)
        match = False
        for g in list(s.empty_groups()):
            if gid == g:
                match = True
                continue
            s.delete_group(g)
        assert len(s.empty_groups()) in (0, 1)
        for _ in xrange(nonempty - (1 if match else 0)):
            s.create_group(rng)
        assert len(s.empty_groups()) == m
        idmap, scores = s.score_value(ei, rng)
        gid = idmap[sample_discrete_log(scores)]
        s.add_value(gid, ei, rng)

def hp(s, hparams, rng=None):
    """
    hparams: dict mapping component id to the following dict:
        {
            'hpdf' : <a function taking values of `hgrid' below and returning scores>,
            'hgrid': <a grid of hyperparameters to try>
        }
    """
    for fi, hparam in hparams.iteritems():
        hpdf, hgrid = hparam['hpdf'], hparam['hgrid']
        scores = np.zeros(len(hgrid))
        for i, hp in enumerate(hgrid):
            s.set_component_hp(fi, hp)
            scores[i] = hpdf(hp) + s.score_likelihood(fi, rng)
        choice = sample_discrete_log(scores)
        s.set_component_hp(fi, hgrid[choice])
