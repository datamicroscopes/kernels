import argparse
#import numpy as np
import itertools as it
import time
import math
import json
#import matplotlib.pylab as plt

from datetime import datetime
from microscopes.common.rng import rng
from microscopes.kernels import gibbs
from vendor import cpuinfo


def versions():
    from microscopes.common import __version__ as common_version
    from microscopes.mixture import __version__ as mixturemodel_version
    from microscopes.kernels import __version__ as kernels_version

    def parse(version):
        tag = version.split('.')[-1]
        print tag
        if len(tag) <= 10:
            # sha1 tags are probably longer than 10 chars
            return None
        if '-' not in tag:
            return tag
        sha1 = tag.split('-')[0][:8]
        if 'debug' in tag:
            return sha1 + 'debug'
        return sha1
    return {
        'common': parse(common_version),
        'mixturemodel': parse(mixturemodel_version),
        'kernels': parse(kernels_version),
    }


def measure(groups, target_runtime, latent, r):
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
        time_per_iter = (cur - loop_start) / float(iters_before_check)
        remaining = target_runtime - elapsed
        iters_before_check = max(int(math.ceil(remaining / time_per_iter)), 1)
        loop_start = cur

    time_per_iteration = elapsed / float(iters)
    return time_per_iteration


def bench(args, latent_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, action='append')
    parser.add_argument('--entities-per-group', type=int, action='append')
    parser.add_argument('--features', type=int, action='append')
    parser.add_argument('--target-runtime', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args(args)

    print args

    if not args.groups or not len(args.groups):
        raise ValueError("need to specify >= 1 --groups")
    if len(args.groups) == 1:
        print "WARNING: one group will make very uninteresting graphs"
    for groups in args.groups:
        if groups <= 0:
            raise ValueError('need positive groups')

    if not args.entities_per_group or not len(args.entities_per_group):
        raise ValueError("need to specify >= 1 --entities-per-group")
    for entities_per_group in args.entities_per_group:
        if entities_per_group <= 0:
            raise ValueError('need positive entities_per_group')

    if not args.features or not len(args.features):
        raise ValueError("need to specify >= 1 --features")
    for features in args.features:
        if features <= 0:
            raise ValueError('need positive features')

    if args.target_runtime <= 0:
        raise ValueError("--target-runtime needs to be >= 0")

    vs = versions()
    vstr = 'c{}-m{}-k{}'.format(vs['common'],
                                vs['mixturemodel'], vs['kernels'])
    print 'vstr:', vstr

    r = rng()

    target_runtime = args.target_runtime
    results = []
    grid = it.product(args.groups, args.entities_per_group, args.features)
    for groups, entities_per_group, features in grid:
        start = time.time()
        latent = latent_fn(groups, entities_per_group, features, r)
        results.append(measure(groups, target_runtime, latent, r))
        print 'finished ({}, {}, {}) in {} seconds'.format(
            groups, entities_per_group, features, time.time() - start)

    output = {
        'args': args.__dict__,
        'versions': vs,
        'cpuinfo': cpuinfo.get_cpu_info(),
        'results': results,
        'time': datetime.now().isoformat(),
    }

    with open(args.output, 'w') as fp:
        json.dump(output, fp)

    # results = np.array(results).reshape(
    #    (len(args.groups), len(args.entities_per_group), len(args.features)))
    #groups = np.array(args.groups, dtype=np.float)
    # for i in xrange(len(args.features)):
    #    data = results[:, :, i]
    #    linear = groups * \
    #        (data[0, 0] /
    #         (float(args.entities_per_group[0]) * groups[0]) / groups[0])
    #    plt.plot(args.groups, linear, 'k--')
    #    for j in xrange(len(args.entities_per_group)):
    #        plt.plot(
    #            args.groups,
    #            data[:, j] / (float(args.entities_per_group[j]) * groups))
    #    legend = ['linear']
    #    legend.extend(['gsize {}'.format(gsize)
    #                   for gsize in args.entities_per_group])
    #    plt.legend(legend, loc='lower right')
    #    plt.xlabel('groups')
    #    plt.ylabel('time/iteration/entity (sec)')
    #    plt.ylim(ymin=0)
    #    plt.tight_layout()
    #    plt.savefig(
    #        '{}-perf-{}-f{}.pdf'.format(prefix, vstr, args.features[i]))
    #    plt.close()
