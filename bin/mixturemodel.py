from microscopes.mixture.definition import model_definition
from microscopes.models import bb

from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.mixture.model import bind, initialize

import numpy as np
import itertools as it
import sys

from bench import bench


def latent(groups, entities_per_group, features, r):
    N = groups * entities_per_group
    defn = model_definition(N, [bb] * features)

    # generate fake data
    Y = np.random.random(size=(N, features)) <= 0.5
    view = numpy_dataview(
        np.array([tuple(y) for y in Y], dtype=[('', bool)] * features))

    # assign entities to their respective groups
    assignment = [[g] * entities_per_group for g in xrange(groups)]
    assignment = list(it.chain.from_iterable(assignment))

    latent = bind(initialize(defn, view, r, assignment=assignment), view)
    latent.create_group(r)  # perftest() doesnt modify group assignments

    return latent

if __name__ == '__main__':
    bench(sys.argv[1:], latent)
