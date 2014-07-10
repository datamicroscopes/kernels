from microscopes.cxx.kernels._bootstrap_h cimport likelihood as c_likelihood

from microscopes.cxx.mixture._model cimport state
from microscopes.cxx.common.recarray._dataview cimport abstract_dataview
from microscopes.cxx.common._rng cimport rng

def likelihood(state mm, abstract_dataview view, rng r):
    c_likelihood(mm._thisptr.get()[0], view._thisptr.get()[0], r._thisptr[0])
