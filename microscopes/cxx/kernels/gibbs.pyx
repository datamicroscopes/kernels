# cimports
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stddef cimport size_t

from microscopes.cxx.kernels._gibbs_h cimport \
        assign_fixed as c_assign_fixed, \
        assign as c_assign, \
        assign_resample as c_assign_resample, \
        hp as c_hp, \
        grid_t
from microscopes.cxx.common._entity_state cimport \
        fixed_entity_based_state_object, \
        entity_based_state_object
from microscopes.cxx.common._rng cimport rng
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t
from microscopes.cxx._models_h cimport hypers_shared_ptr, hypers_raw_ptr
from microscopes.cxx._models cimport _base

# python imports
from microscopes.cxx._models import _base

def assign_fixed(fixed_entity_based_state_object s, rng r):
    assert r
    c_assign_fixed(s._thisptr.get()[0], r._thisptr[0])

def assign(entity_based_state_object s, rng r):
    assert r
    c_assign(s.px().get()[0], r._thisptr[0])

def assign_resample(entity_based_state_object s, int m, rng r):
    assert r
    c_assign_resample(s.px().get()[0], m, r._thisptr[0])

def hp(fixed_entity_based_state_object s, dict params, rng r):
    assert r
    cdef vector[pair[size_t, grid_t]] g
    cdef grid_t g0
    cdef vector[hypers_shared_ptr] ptrs
    cdef hyperparam_bag_t raw
    for fi, ps in params.iteritems():
        g0.clear()
        prior_fn, grid = ps['hpdf'], ps['hgrid']
        for p in grid:
            prior_score = prior_fn(p)
            c_desc = s._models[fi].c_desc()
            if not isinstance(c_desc, _base):
                raise RuntimeError(
                    "expecting _base, got {}".format(repr(c_desc)))
            ptrs.push_back((<_base>c_desc).create_hypers())
            raw = s._models[fi].py_desc().shared_dict_to_bytes(p)
            ptrs.back().get().set_hp(raw)
            g0.push_back(
                pair[hypers_raw_ptr, float](ptrs.back().get(), prior_score))
        g.push_back(pair[size_t, grid_t](fi, g0))
    c_hp(s._thisptr.get()[0], g, r._thisptr[0])
