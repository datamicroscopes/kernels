from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.cxx.kernels._slice_h cimport hp as c_hp, slice_t, sample as c_sample

from microscopes.cxx.mixture._model cimport state
from microscopes.cxx.common._scalar_functions cimport scalar_function
from microscopes.cxx.common._typedefs_h cimport scalar_1d_float_fn
from microscopes.cxx.common._rng cimport rng

def hp(state s, dict params, rng r):
    cdef vector[pair[size_t, slice_t]] c_params
    cdef slice_t c_slice
    for fi, p in params.iteritems():
        c_slice.clear()
        for k, v in p.iteritems():
            scorefn, w = v
            assert isinstance(scorefn, scalar_function), 'functions must be scalar_functions'
            w = float(w)
            c_slice[k] = pair[scalar_1d_float_fn, float]((<scalar_function>scorefn)._func, w)
        c_params.push_back(pair[size_t, slice_t](fi, c_slice))
    c_hp(s._thisptr[0], c_params, r._thisptr[0])

def sample(scalar_function func, float x0, float w, rng r):
    return c_sample(func._func, x0, w, r._thisptr[0])
