"""
Slice sampler based on:
    http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    https://github.com/ericmjonas/netmotifs/blob/master/irm/slicesample.cc
"""

import numpy as np

import re
import math
import logging
logger = logging.getLogger(__name__)

def interval(pdf, x0, y, w, m):
    """
    Fig. 3 of http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    """
    U = np.random.random()
    L = x0 - w*U
    R = L + w
    V = np.random.random()
    J = int(math.floor(m*V))
    K = m-1-J

    while J > 0 and y < pdf(L):
        L -= w
        J -= 1

    while K > 0 and y < pdf(R):
        R += w
        K -= 1

    if J == 0 or K == 0:
        logging.warn('interval hit maximum expansions')
    return L, R

def shrink(pdf, x0, y, L, R, ntries):
    """
    Fig. 5 of http://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
    """
    while ntries:
        U = np.random.random()
        x1 = L + U*(R-L)
        if y < pdf(x1):
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1
        ntries -= 1

    logging.warn('shrink exceeded maximum iterations (%d)' % (ntries))
    return x1

def sample(pdf, x0, w, r=None):
    y = np.log(np.random.random()) + pdf(x0)
    L, R = interval(pdf, x0, y, w, m=10000)
    return shrink(pdf, x0, y, L, R, ntries=100)

_desc_regex = re.compile(r'(.+)\[(\d+)\]$')
def _parse_descriptor(desc, default=None):
    m = _desc_regex.match(desc)
    if not m:
        return desc, default
    return m.group(1), int(m.group(2))

def hp(s, r=None, cparam=None, hparams=None):
    # XXX: None should indicate use a sane default
    if cparam is None:
        cparam = {}
    if hparams is None:
        hparams = {}

    def get_hp_value(update_desc):
        key, idx = update_desc
        val = hp[key]
        if idx is not None:
            val = val[idx]
        return val

    def set_hp_value(update_desc, val):
        key, idx = update_desc
        if idx is not None:
            hp[key][idx] = val
        else:
            hp[key] = val

    # XXX: fix code dup

    hp = s.get_cluster_hp()
    for update_descs, (prior, w) in cparam.iteritems():
        if not hasattr(update_descs, '__iter__'):
            update_descs = [update_descs]
        #if len(update_descs) != prior.input_dim():
        #    raise ValueError("wrong # of args for prior function")
        update_descs = map(_parse_descriptor, update_descs)
        args = map(get_hp_value, update_descs)
        for argpos, update_desc in enumerate(update_descs):
            def scorefn(x):
                set_hp_value(update_desc, x)
                args[argpos] = x
                s.set_cluster_hp(hp)
                return prior(*args) + s.score_assignment()
            samp = sample(scorefn, args[argpos], w)
            set_hp_value(update_desc, samp)
            args[argpos] = samp
    s.set_cluster_hp(hp)

    for fi, hparam in hparams.iteritems():
        hp = s.get_component_hp(fi)
        for update_descs, (prior, w) in hparam.iteritems():
            if not hasattr(update_descs, '__iter__'):
                update_descs = [update_descs]
            #if len(update_descs) != prior.input_dim():
            #    raise ValueError("wrong # of args for prior function")
            update_descs = map(_parse_descriptor, update_descs)
            args = map(get_hp_value, update_descs)
            for argpos, update_desc in enumerate(update_descs):
                def scorefn(x):
                    set_hp_value(update_desc, x)
                    args[argpos] = x
                    s.set_component_hp(fi, hp)
                    return prior(*args) + s.score_likelihood(fi, r)
                samp = sample(scorefn, args[argpos], w)
                set_hp_value(update_desc, samp)
                args[argpos] = samp
        s.set_component_hp(fi, hp)

def theta(s, r=None, tparams=None):
    # XXX: None should indicate use a sane default
    if tparams is None:
        tparams = {}

    for fi, params in tparams.iteritems():
        groups = np.array(s.suffstats_identifiers(fi), dtype=np.int)
        for k, w in params.iteritems():
            for gi in groups[np.random.permutation(len(groups))]:
                theta = s.get_suffstats(fi, gi)
                def pdf(x):
                    theta[k] = x
                    s.set_suffstats(fi, gi, theta)
                    return s.score_likelihood_indiv(fi, gi, r)
                theta[k] = slice_sample(pdf, theta[k], w)
