from microscopes.cxx.mixture._model_h cimport state
from microscopes.cxx.common._dataview_h cimport dataview
from microscopes.cxx.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/kernels/bootstrap.hpp" namespace "microscopes::kernels::bootstrap":
    void likelihood(state &, dataview &, rng_t &) except +
