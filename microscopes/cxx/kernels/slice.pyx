from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.cxx.kernels._slice_h cimport \
        hp as c_hp, \
        theta as c_theta, \
        slice_hp_param_component_t, \
        slice_hp_param_t, \
        slice_hp_t, \
        slice_theta_param_t, \
        slice_theta_t, \
        sample as c_sample

from microscopes.cxx.mixture._model cimport state
from microscopes.cxx.common._scalar_functions cimport scalar_function
from microscopes.cxx.common._typedefs_h cimport scalar_1d_float_fn
from microscopes.cxx.common._rng cimport rng

def sample(scalar_function func, float x0, float w, rng r):
    return c_sample(func._func, x0, w, r._thisptr[0])

def hp(state s, dict cparams, dict hparams, rng r):
    cdef vector[slice_hp_param_t] c_cparams 
    cdef vector[slice_hp_t] c_hparams

    cdef vector[slice_hp_param_component_t] buf0
    cdef vector[slice_hp_param_t] buf1

    for k, objs in cparams.iteritems():
        buf0.clear()
        if not hasattr(objs, '__iter__'):
            objs = [objs]
        for param in objs:
            assert isinstance(param._prior, scalar_function)
            buf0.push_back(
                slice_hp_param_component_t(
                    0 if param.index() is None else param.index(), 
                    (<scalar_function>param._prior)._func,
                    param._w))
        c_cparams.push_back(slice_hp_param_t(k, buf0))

    for fi, hparam in hparams.iteritems():
        buf1.clear()
        for k, objs in hparam.iteritems():
            buf0.clear()
            # XXX: code duplication with above
            if not hasattr(objs, '__iter__'):
                objs = [objs]
            for param in objs:
                assert isinstance(param._prior, scalar_function)
                buf0.push_back(
                    slice_hp_param_component_t(
                        0 if param.index() is None else param.index(), 
                        (<scalar_function>param._prior)._func,
                        param._w))
            buf1.push_back(slice_hp_param_t(k, buf0))
        c_hparams.push_back(slice_hp_t(fi, buf1))

    c_hp(s._thisptr[0], c_cparams, c_hparams, r._thisptr[0])

def theta(state s, dict tparams, rng r):
    cdef vector[slice_theta_t] c_tparams
    cdef vector[slice_theta_param_t] buf0
    for fi, params in tparams.iteritems():
        buf0.clear()
        for k, w in params.iteritems():
            buf0.push_back(slice_theta_param_t(k, w))
        c_tparams.push_back(slice_theta_t(fi, buf0))
    c_theta(s._thisptr[0], c_tparams, r._thisptr[0])
