from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stddef cimport size_t

from microscopes.cxx.mixture._model_h cimport state
from microscopes.cxx.common._dataview_h cimport dataview
from microscopes.cxx.common._random_fwd_h cimport rng_t
from microscopes.cxx._models_h cimport model_raw_ptr

cdef extern from "microscopes/kernels/gibbs.hpp" namespace "microscopes::kernels::gibbs":
    ctypedef vector[pair[model_raw_ptr, float]] grid_t
    void hp(state &, vector[pair[size_t, grid_t]] &, rng_t &) except +
    void assign(state &, dataview &, rng_t &) except +
    void assign_resample(state &, dataview &, size_t, rng_t &) except +
