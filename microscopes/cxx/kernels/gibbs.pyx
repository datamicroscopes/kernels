from microscopes.cxx.kernels._gibbs_h cimport assign as c_assign

from microscopes.cxx.mixture._model cimport state
from microscopes.cxx.common._dataview cimport abstract_dataview
from microscopes.cxx.common._rng cimport rng

def assign(state mm, abstract_dataview view, rng r):
    c_assign(mm._thisptr[0], view._thisptr[0], r._thisptr[0])
