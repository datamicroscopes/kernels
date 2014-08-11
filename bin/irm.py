from microscopes.irm.definition import model_definition
from microscopes.models import bb

from microscopes.common.relation.dataview import numpy_dataview
from microscopes.irm.model import bind, initialize

import numpy as np
import itertools as it
import sys

from bench import bench

# features = relations here


def latent(groups, entities_per_group, features, r):
    N = groups * entities_per_group
    defn = model_definition([N], [((0, 0), bb)] * features)

    # generate fake data
    views = []
    for i in xrange(features):
        Y = np.random.random(size=(N, N)) <= 0.5
        view = numpy_dataview(Y)
        views.append(view)

    # assign entities to their respective groups
    assignment = [[g] * entities_per_group for g in xrange(groups)]
    assignment = list(it.chain.from_iterable(assignment))

    latent = bind(
        initialize(defn, views, r, domain_assignments=[assignment]), 0, views)
    latent.create_group(r)  # perftest() doesnt modify group assignments

    return latent

if __name__ == '__main__':
    bench(sys.argv[1:], latent, 'irm')
