from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.map cimport map
from libc.stddef cimport size_t

from microscopes.cxx.common._entity_state_h cimport fixed_entity_based_state_object 
from microscopes.cxx.common._typedefs_h cimport scalar_1d_float_fn
from microscopes.cxx.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/kernels/slice.hpp" namespace "microscopes::kernels::slice":
    float sample(scalar_1d_float_fn fn, float, float, rng_t &) except +

    cdef cppclass slice_hp_param_component_t:
        slice_hp_param_component_t()
        slice_hp_param_component_t(size_t, scalar_1d_float_fn, float)

    cdef cppclass slice_hp_param_t:
        slice_hp_param_t()
        slice_hp_param_t(string &, vector[slice_hp_param_component_t] &) except +

    cdef cppclass slice_hp_t:
        slice_hp_t()
        slice_hp_t(size_t, vector[slice_hp_param_t] &) except +

    cdef cppclass slice_theta_param_t:
        slice_theta_param_t()
        slice_theta_param_t(string &, float) except +

    cdef cppclass slice_theta_t:
        slice_theta_t()
        slice_theta_t(size_t, vector[slice_theta_param_t] &) except +

    void hp(fixed_entity_based_state_object &, vector[slice_hp_param_t] &, vector[slice_hp_t] &, rng_t &) except +
    void theta(fixed_entity_based_state_object &, vector[slice_theta_t] &, rng_t &) except +

