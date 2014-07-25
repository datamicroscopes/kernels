import argparse
import numpy as np
import itertools as it
import time
import math
import matplotlib.pylab as plt

from microscopes.mixture.definition import model_definition
from microscopes.models import bb
from microscopes.cxx.common.rng import rng
from microscopes.cxx.kernels import gibbs
from microscopes.cxx.common.recarray.dataview import numpy_dataview
from microscopes.cxx.mixture.model import bind, initialize


def measure(groups, entities_per_group, features, target_runtime):
    if groups <= 0:
        raise ValueError('need positive groups')
    if entities_per_group <= 0:
        raise ValueError('need positive entities_per_group')
    if features <= 0:
        raise ValueError('need positive features')
    if target_runtime <= 0:
        raise ValueError('need positive target runtime')

    N = groups * entities_per_group
    defn = model_definition(N, [bb]*features)

    # generate fake data
    Y = np.random.random(size=(N, features)) <= 0.5
    view = numpy_dataview(np.array([tuple(y) for y in Y], dtype=[('', bool)]*features))

    # assign entities to their respective groups
    assignment = [[g]*entities_per_group for g in xrange(groups)]
    assignment = list(it.chain.from_iterable(assignment))

    r = rng()

    latent = bind(initialize(defn, view, r, assignment=assignment), view)
    latent.create_group(r) # perftest() doesnt modify group assignments
    start = time.time()
    loop_start = start
    iters = 0
    iters_before_check = 1
    while 1:
        for _ in xrange(iters_before_check):
            gibbs.perftest(latent, r)
        iters += iters_before_check
        cur = time.time()
        elapsed = cur - start
        if elapsed >= target_runtime:
            break
        time_per_iter = (cur - loop_start)/float(iters_before_check)
        remaining = target_runtime - elapsed
        iters_before_check = max(int(math.ceil(remaining / time_per_iter)), 1)
        loop_start = cur

    time_per_iteration = elapsed / float(iters)
    return time_per_iteration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, action='append')
    parser.add_argument('--entities-per-group', type=int, action='append')
    parser.add_argument('--features', type=int, action='append')
    parser.add_argument('--target-runtime', type=int, required=True)
    args = parser.parse_args()
    print args
    target_runtime = args.target_runtime
    results = []
    for groups, entities_per_group, features in it.product(args.groups, args.entities_per_group, args.features):
        start = time.time()
        results.append(measure(groups, entities_per_group, features, target_runtime))
        print 'finished ({}, {}, {}) in {} seconds'.format(groups, entities_per_group, features, time.time()-start)

    results = np.array(results).reshape((len(args.groups), len(args.entities_per_group), len(args.features)))
    for i in xrange(len(args.features)):
        data = results[:,:,i]
        for j in xrange(len(args.entities_per_group)):
            plt.plot(args.groups, data[:,j])
        plt.legend(['gsize {}'.format(gsize) for gsize in args.entities_per_group])
        plt.xlabel('groups')
        plt.ylabel('time/iteration (sec)')
        plt.savefig('mixturemodel-perf-f{}.pdf'.format(args.features[i]))
        plt.close()

