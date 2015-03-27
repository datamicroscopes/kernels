from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stddef cimport size_t

from microscopes.common._entity_state_h cimport entity_based_state_object
from microscopes.common._random_fwd_h cimport rng_t
from microscopes._models_h cimport hypers_raw_ptr

cdef extern from "microscopes/kernels/gibbs.hpp" namespace "microscopes::kernels::gibbs":
    ctypedef vector[pair[hypers_raw_ptr, float]] grid_t
    void assign(entity_based_state_object &, rng_t &) except +
    void assign_resample(entity_based_state_object &, size_t, rng_t &) except +
    void hp(entity_based_state_object &, vector[pair[size_t, grid_t]] &, rng_t &) except +
    void perftest(entity_based_state_object &, rng_t &) except +
