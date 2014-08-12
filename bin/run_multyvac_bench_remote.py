import argparse
import os

from microscopes.common.util import mkdirp

import mixturemodel
import irm
from bench import bench

# XXX: racy


def get_next_id(d):
    fnames = os.listdir(d)

    def parse(fname):
        toks = fname.split(".")
        if len(toks) != 2 or toks[1] != 'json':
            return None
        try:
            return int(toks[0])
        except ValueError:
            return None

    fnames = map(parse, fnames)
    fnames = [fname for fname in fnames if fname is not None]
    return max(fnames) + 1 if len(fnames) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--benchmark', required=True)
    args = parser.parse_args()

    benchmarks = {
        'mixturemodel': (mixturemodel.latent, {
            '--groups': range(10, 101, 10),
            '--entities-per-group': [10, 100],
            '--features': 10,
            '--target-runtime': 10,
        }),
        'irm': (irm.latent, {
            '--groups': range(10, 101, 10),
            '--entities-per-group': [10],
            '--features': 1,
            '--target-runtime': 10,
        }),
    }

    if args.benchmark not in benchmarks:
        raise ValueError(
            "invalid benchmark: {}".format(args.benchmark))

    mkdirp(args.results_dir)

    def format_args(args):
        toks = []
        for k, v in args.iteritems():
            if hasattr(v, '__iter__'):
                for v0 in v:
                    toks.extend([k, str(v0)])
            else:
                toks.extend([k, str(v)])
        return toks

    latent, benchargs = benchmarks[args.benchmark]
    d = os.path.join(args.results_dir, args.benchmark)
    mkdirp(d)
    nextid = get_next_id(d)
    benchargs = format_args(benchargs)
    benchargs.extend([
        '--output',
        os.path.join(d, "{id}.json".format(id=nextid))])
    bench(benchargs, latent)

if __name__ == '__main__':
    main()
