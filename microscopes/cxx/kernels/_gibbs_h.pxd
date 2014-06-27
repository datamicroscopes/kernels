from microscopes.cxx.mixture._model_h cimport state
from microscopes.cxx.common._dataview_h cimport dataview
from microscopes.cxx.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/kernels/gibbs.hpp" namespace "microscopes::kernels::gibbs":
    void assign(state &, dataview &, rng_t &) except +
