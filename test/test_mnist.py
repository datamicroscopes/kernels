from microscopes.cxx.common.dataview import numpy_dataview
from microscopes.cxx.common.rng import rng
from microscopes.cxx.common.scalar_functions import log_exponential
from microscopes.cxx.models import bb
from microscopes.cxx.mixture.model import state
from microscopes.cxx.kernels.gibbs import assign
from microscopes.cxx.kernels.slice import hp
from microscopes.cxx.kernels.bootstrap import likelihood

from sklearn.datasets import fetch_mldata
mnist_dataset = fetch_mldata('MNIST original')

import numpy as np
import numpy.ma as ma
import matplotlib.pylab as plt
import math
import time

from nose.plugins.attrib import attr

def make_dp(n, models, clusterhp, featurehps):
    s = state(n, models)
    s.set_cluster_hp(clusterhp)
    for i, hp in enumerate(featurehps):
        s.set_feature_hp(i, hp)
    return s

def groupcounts(s):
    counts = np.zeros(s.ngroups(), dtype=np.int)
    for i, gid in enumerate(s.groups()):
        counts[i] = s.groupsize(gid)
    return np.sort(counts)[::-1]

def groupsbysize(s):
    """groupids by decreasing size"""
    counts = [(gid, s.groupsize(gid)) for gid in s.groups()]
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    return [i for i, _ in counts]

@attr('slow')
def test_mnist():
    Y_2 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 2.)[0]]
    Y_3 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 3.)[0]]
    print 'number of twos:', Y_2.shape[0]
    print 'number of threes:', Y_3.shape[0]
    _, D = Y_2.shape
    W = int(math.sqrt(D))
    assert W * W == D
    dtype = [('', bool)]*D
    Y = np.vstack([Y_2, Y_3])
    Y = np.array([tuple(y) for y in Y[np.random.permutation(np.arange(Y.shape[0]))]], dtype=dtype)

    view = numpy_dataview(Y)
    s = make_dp(Y.shape[0], [bb]*D, {'alpha':0.2}, [{'alpha':1.,'beta':1.}]*D)

    r = rng()
    likelihood(s, view.view(False, r), r)

    indiv_prior_fn = log_exponential(1.2)
    hparams = {
        i : {
            'alpha' : (indiv_prior_fn, 1.5),
            'beta'  : (indiv_prior_fn, 1.5),
        } for i in xrange(D) }

    def plot(s, fname):
        hps = [s.get_feature_hp(i) for i in xrange(D)]
        def prior_prob(hp):
            return hp['alpha'] / (hp['alpha'] + hp['beta'])
        def data_for_group(gid):
            suffstats = [s.get_suff_stats(gid, i) for i in xrange(D)]
            def prob(hp, ss):
                top = hp['alpha'] + ss['heads']
                bot = top + hp['beta'] + ss['tails']
                return top/bot
            probs = [prob(hp, ss) for hp, ss in zip(hps, suffstats)]
            return np.array(probs)
        data = [data_for_group(g) for g in groupsbysize(s)]
        digits_per_row = 12
        rem = len(data) % digits_per_row
        if rem:
            fill = digits_per_row - rem
            for _ in xrange(fill):
                data.append(np.zeros(D))
        assert not (len(data) % digits_per_row)
        rows = len(data) / digits_per_row
        data = np.vstack([np.hstack([d.reshape((W,W)) for d in data[i:i+digits_per_row]]) for i in xrange(0, len(data), digits_per_row)])
        #print 'saving figure', fname
        plt.imshow(data, cmap=plt.cm.binary, interpolation='nearest')
        plt.savefig(fname)
        plt.axes().set_aspect('equal')
        plt.close()

    def kernel(rid):
        start = time.time()
        assign(s, view.view(True, r), r)
        hp(s, hparams, r)
        sec = time.time() - start
        print 'rid=', rid, 'nclusters=', s.ngroups(), 'iter=', sec, 'sec'

    # burnin
    burnin = 30
    for rid in xrange(burnin):
        kernel(rid)
    print 'finished burnin'
    plot(s, 'mnist_clusters.pdf')

    # posterior inference
    present = D/2
    absent = D-present
    queries = [tuple(Y[i]) for i in np.random.permutation(Y.shape[0])[:8]]
    queries = ma.masked_array(
        np.array(queries, dtype=[('',bool)]*D),
        mask=[(False,)*present + (True,)*absent])

    def postpred_sample(y_new):
        Y_samples = [s.sample_post_pred(y_new, r)[1] for _ in xrange(1000)]
        Y_samples = np.array([list(y) for y in np.hstack(Y_samples)])
        Y_avg = Y_samples.mean(axis=0)
        return Y_avg

    queries = [postpred_sample(y) for y in queries]
    data = np.hstack([q.reshape((W,W)) for q in queries])
    plt.imshow(data, cmap=plt.cm.binary, interpolation='nearest')
    plt.savefig('mnist_predict.pdf')
    plt.close()
