# XXX(stephentu):
# Would be nice to move this test into the common repo, but mixturemodel/IRM is
# our only python level API to the CRP and the kernels test folder has all
# sorts of nice permutation utilities (which should be moved to a separate util
# file)

from microscopes.py.mixture.model import \
        initialize as py_initialize
from microscopes.py.common.recarray.dataview import \
        numpy_dataview as py_numpy_dataview
from microscopes.cxx.mixture.model import \
        initialize as cxx_initialize
from microscopes.cxx.common.recarray.dataview import \
        numpy_dataview as cxx_numpy_dataview

from microscopes.cxx.common.rng import rng
from microscopes.models import bb
from microscopes.mixture.definition import model_definition

from test_utils import permutation_iter
from nose.tools import assert_almost_equals

import itertools as it
import math
import numpy as np

def _test_crp(initialize_fn, dataview, alpha, r):
    N = 6
    defn = model_definition(N, [bb])
    Y = np.array([(True,)]*N, dtype=[('',bool)])
    view = dataview(Y)
    def crp_score(assignment):
        latent = initialize_fn(
            defn, view, r=r,
            cluster_hp={'alpha':alpha}, assignment=assignment)
        return latent.score_assignment()
    dist = np.array(list(map(crp_score, permutation_iter(N))))
    dist = np.exp(dist)
    assert_almost_equals(dist.sum(), 1.0, places=3)

def test_crp_py():
    for alpha in (0.1, 1.0, 10.0):
        _test_crp(py_initialize, py_numpy_dataview, alpha=alpha, r=None)

def test_crp_cxx():
    for alpha in (0.1, 1.0, 10.0):
        _test_crp(cxx_initialize, cxx_numpy_dataview, alpha=alpha, r=rng())
