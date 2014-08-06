from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.common._entity_state_h cimport fixed_entity_based_state_object 
from microscopes.common._scalar_functions_h cimport scalar_fn
from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/kernels/slice.hpp" namespace "microscopes::kernels::slice":
    float sample_1d(scalar_fn fn, float, float, rng_t &) except +

    cdef cppclass slice_update_param_t:
        slice_update_param_t()
        slice_update_param_t(const string &, size_t) except +

    cdef cppclass slice_hp_param_t:
        slice_hp_param_t()
        slice_hp_param_t(const vector[slice_update_param_t] &, 
                         scalar_fn,
                         float) except +

    cdef cppclass slice_hp_t:
        slice_hp_t()
        slice_hp_t(size_t, const vector[slice_hp_param_t] &) except +

    cdef cppclass slice_theta_param_t:
        slice_theta_param_t()
        slice_theta_param_t(string &, float) except +

    cdef cppclass slice_theta_t:
        slice_theta_t()
        slice_theta_t(size_t, vector[slice_theta_param_t] &) except +

    void hp(fixed_entity_based_state_object &, 
            const vector[slice_hp_param_t] &, 
            const vector[slice_hp_t] &, 
            rng_t &) except +

    void theta(fixed_entity_based_state_object &, 
               const vector[slice_theta_t] &, 
               rng_t &) except +
