# cython imports
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.kernels._slice_h cimport \
    hp as c_hp, \
    theta as c_theta, \
    slice_update_param_t, \
    slice_hp_param_t, \
    slice_hp_t, \
    slice_theta_param_t, \
    slice_theta_t, \
    sample_1d as c_sample_1d

from microscopes.common._entity_state cimport \
    fixed_entity_based_state_object
from microscopes.common._scalar_functions cimport scalar_function
from microscopes.common._rng cimport rng

# python imports
from microscopes.common import validator
import re

_desc_regex = re.compile(r'(.+)\[(\d+)\]$')


def _parse_descriptor(desc, default=None):
    m = _desc_regex.match(desc)
    if not m:
        return desc, default
    return m.group(1), int(m.group(2))


def sample(scalar_function func, float x0, float w, rng r):
    validator.validate_not_none(r, "r")
    return c_sample_1d(func._func, x0, w, r._thisptr[0])


def hp(fixed_entity_based_state_object s, rng r, cparam=None, hparams=None):
    """

    example invocation:

    hparams = {
      0 : {
            ('alpha', 'beta') : (log_noninformative_beta_prior, 0.1),
          },
      1 : {
            'alphas[0]' : (log_exponential(1), 0.1),
            'alphas[1]' : (log_exponential(1), 0.1),
          },
    }
    hp(s, None, hparams, r)
    """
    validator.validate_not_none(r, "r")

    # XXX: None should indicate use a sane default
    if cparam is None:
        cparam = {}
    if hparams is None:
        hparams = {}

    cdef vector[slice_hp_param_t] c_cparam
    cdef vector[slice_hp_t] c_hparams

    cdef vector[slice_hp_param_t] buf0
    cdef vector[slice_update_param_t] buf1

    for update_descs, (prior, w) in cparam.iteritems():
        if not hasattr(update_descs, '__iter__'):
            update_descs = [update_descs]
        buf1.clear()
        for update_desc in update_descs:
            key, idx = _parse_descriptor(update_desc, default=0)
            buf1.push_back(slice_update_param_t(key, idx))
        validator.validate_type(prior, scalar_function)
        validator.validate_positive(w)
        c_cparam.push_back(
            slice_hp_param_t(
                buf1,
                (<scalar_function>prior)._func,
                w))

    for fi, hparam in hparams.iteritems():
        buf0.clear()
        for update_descs, (prior, w) in hparam.iteritems():
            if not hasattr(update_descs, '__iter__'):
                update_descs = [update_descs]
            buf1.clear()
            for update_desc in update_descs:
                key, idx = _parse_descriptor(update_desc, default=0)
                buf1.push_back(slice_update_param_t(key, idx))
            validator.validate_type(prior, scalar_function)
            validator.validate_positive(w)
            buf0.push_back(
                slice_hp_param_t(
                    buf1,
                    (<scalar_function>prior)._func,
                    w))
        c_hparams.push_back(slice_hp_t(fi, buf0))

    c_hp(s._thisptr.get()[0], c_cparam, c_hparams, r._thisptr[0])


def theta(fixed_entity_based_state_object s, rng r, tparams=None):
    validator.validate_not_none(r, "r")
    # XXX: None should indicate use a sane default
    if tparams is None:
        tparams = {}
    cdef vector[slice_theta_t] c_tparams
    cdef vector[slice_theta_param_t] buf0
    for fi, params in tparams.iteritems():
        buf0.clear()
        for k, w in params.iteritems():
            buf0.push_back(slice_theta_param_t(k, w))
        c_tparams.push_back(slice_theta_t(fi, buf0))
    c_theta(s._thisptr.get()[0], c_tparams, r._thisptr[0])
