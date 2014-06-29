from distributions.dbg.random import sample_discrete_log

import numpy as np

def likelihood(s, it, r=None):
    assert not s.ngroups()
    assert (np.array(s.assignments(), dtype=np.int)==-1).all()

    ei0, y0 = next(it)
    gid0 = s.create_group(r)
    s.add_value(gid0, ei0, y0, r)
    empty_gid = s.create_group(r)
    for ei, yi in it:
        idmap, scores = s.score_value(yi, r)
        gid = idmap[sample_discrete_log(scores)]
        s.add_value(gid, ei, yi, r)
        if gid == empty_gid:
            empty_gid = s.create_group(r)

    assert not (np.array(s.assignments(), dtype=np.int)==-1).any()
    return s
