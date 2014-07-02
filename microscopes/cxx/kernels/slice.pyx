from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.cxx.kernels._slice_h cimport hp as c_hp, \
        slice_t, slice_indiv_t, sample as c_sample

from microscopes.cxx.mixture._model cimport state
from microscopes.cxx.common._scalar_functions cimport scalar_function
from microscopes.cxx.common._typedefs_h cimport scalar_1d_float_fn
from microscopes.cxx.common._rng cimport rng

def hp(state s, dict params, rng r):
    cdef vector[pair[size_t, slice_t]] c_params
    cdef slice_t c_slice
    cdef vector[slice_indiv_t] c_slice_indivs
    for fi, p in params.iteritems():
        c_slice.clear()
        for k, objs in p.iteritems():
            c_slice_indivs.clear()
            if not hasattr(objs, '__iter__'):
                objs = [objs]
            for param in objs:
                assert isinstance(param._prior, scalar_function), 'functions must be scalar_function'
                idx = param.index()
                if idx is None:
                    idx = 0
                c_slice_indivs.push_back(
                    slice_indiv_t(idx, (<scalar_function>param._prior)._func, param._w))
            c_slice[k] = c_slice_indivs
        c_params.push_back(pair[size_t, slice_t](fi, c_slice))
    c_hp(s._thisptr[0], c_params, r._thisptr[0])

def sample(scalar_function func, float x0, float w, rng r):
    return c_sample(func._func, x0, w, r._thisptr[0])
