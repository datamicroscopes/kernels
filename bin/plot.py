import sys
import argparse
import os
import json

import numpy as np
import matplotlib.pylab as plt


def draw(obj, outfile):
    groups, entities_per_group, features = (
        obj['args']['groups'],
        obj['args']['entities_per_group'],
        obj['args']['features'],
    )
    results = obj['results']
    results = np.array(results).reshape(
        (len(groups), len(entities_per_group), len(features)))
    groups = np.array(groups, dtype=np.float)
    for i in xrange(len(features)):
        data = results[:, :, i]
        linear = groups * \
            (data[0, 0] /
             (float(entities_per_group[0]) * groups[0]) / groups[0])
        plt.plot(groups, linear, 'k--')
        for j in xrange(len(entities_per_group)):
            plt.plot(
                groups,
                data[:, j] / (float(entities_per_group[j]) * groups))
        legend = ['linear']
        legend.extend(['gsize {}'.format(gsize)
                       for gsize in entities_per_group])
        plt.legend(legend, loc='lower right')
        plt.xlabel('groups')
        plt.ylabel('time/iteration/entity (sec)')
        plt.ylim(ymin=0)
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--sync", action='store_true')
    parser.add_argument("--volume")
    args = parser.parse_args(args)
    if args.sync and not args.volume:
        raise ValueError("--sync requires --volume")
    if args.sync:
        import multyvac
        vol = multyvac.volume.get(args.volume)
        vol.sync_down("", args.results_dir)
    for dirpath, _, filenames in os.walk(args.results_dir):
        for fname in filenames:
            toks = fname.split(".")
            if len(toks) != 2 or toks[1] != 'json':
                continue
            p = os.path.join(dirpath, fname)
            outp = os.path.join(dirpath, '{}.pdf'.format(toks[0]))
            with open(p, 'r') as fp:
                try:
                    obj = json.load(fp)
                except ValueError:
                    print "skipping file {}".format(p)
                    continue
                draw(obj, outp)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
