from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stddef cimport size_t

from microscopes.cxx.kernels._gibbs_h cimport hp as c_hp, assign as c_assign, assign_resample as c_assign_resample, grid_t, model_raw_ptr

from microscopes.cxx.common.recarray._dataview cimport abstract_dataview
from microscopes.cxx.common._rng cimport rng
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t
from microscopes.cxx._models cimport factory
from microscopes.cxx._models_h cimport model_shared_ptr
from microscopes.cxx.mixture._model cimport state

def hp(state s, dict params, rng r):
    assert r
    cdef vector[pair[size_t, grid_t]] g
    cdef grid_t g0
    cdef vector[model_shared_ptr] ptrs
    cdef hyperparam_bag_t raw
    for fi, ps in params.iteritems():
        g0.clear()
        prior_fn, grid = ps['hpdf'], ps['hgrid']
        for p in grid:
            prior_score = prior_fn(p)
            ptrs.push_back((<factory>s._models[fi][1]).new_cmodel())
            raw = s._models[fi][0].shared_dict_to_bytes(p)
            ptrs.back().get().set_hp(raw)
            g0.push_back(pair[model_raw_ptr, float](ptrs.back().get(), prior_score))
        g.push_back(pair[size_t, grid_t](fi, g0))
    c_hp(s._thisptr[0], g, r._thisptr[0])

def assign(state s, abstract_dataview view, rng r):
    assert r
    c_assign(s._thisptr[0], view._thisptr[0], r._thisptr[0])

def assign_resample(state s, abstract_dataview view, int m, rng r):
    assert r
    c_assign_resample(s._thisptr[0], view._thisptr[0], m, r._thisptr[0])
