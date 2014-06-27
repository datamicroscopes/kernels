from distributions.dbg.models import bb
from microscopes.py.common.dataview import numpy_dataview
from microscopes.py.mixture.dp import state
from microscopes.py.kernels.gibbs import gibbs_assign

from sklearn.datasets import fetch_mldata
mnist_dataset = fetch_mldata('MNIST original')

import numpy as np
import numpy.ma as ma
import matplotlib.pylab as plt
import math

from nose.plugins.attrib import attr

def make_dp(n, models, clusterhp, featurehps):
    mm = state(n, models)
    mm.set_cluster_hp(clusterhp)
    for i, hp in enumerate(featurehps):
        mm.set_feature_hp(i, hp)
    return mm

@attr('slow')
def test_mnist():
    Y = mnist_dataset['data'][np.where(mnist_dataset['target'] == 2.)[0]]
    N, D = Y.shape
    W = int(math.sqrt(D))
    assert W * W == D
    dtype = [('', bool)]*D
    N_sampled = min(300, N)
    print 'sampling', N_sampled, 'out of', N
    perm = np.random.permutation(np.arange(N))
    inds = perm[:N_sampled]
    Y_sampled = np.array([tuple(y) for y in Y[inds]], dtype=dtype)

    present = D/2
    absent = D-present
    y_test = tuple(Y[perm[N_sampled]]) # never seen before
    #y_test = tuple(Y[perm[0]])
    y_new = ma.masked_array(
        np.array([y_test], dtype=[('', bool)]*D),
        mask=[(False,)*present + (True,)*absent])[0]

    dataset = numpy_dataview(Y_sampled)
    mm = make_dp(N_sampled, [bb]*D, {'alpha':2.0}, [{'alpha':1.0, 'beta':1.0}]*D)
    mm.bootstrap(dataset.data(shuffle=False))

    burnin_niters = 2000
    for _ in xrange(burnin_niters):
        gibbs_assign(mm, dataset.data(shuffle=True))
    print 'finished burnin of', burnin_niters, 'iters'
    print 'clusters so far:', mm.ngroups()

    def postpred_sample():
        Y_samples = [mm.sample_post_pred(y_new=y_new) for _ in xrange(10000)]
        Y_samples = np.array([list(y) for y in np.hstack(Y_samples)])
        Y_avg = Y_samples.mean(axis=0)
        return Y_avg

    Y_samples = np.array([postpred_sample() for _ in xrange(1)])
    Y_avg = Y_samples.mean(axis=0)

    plt.imshow(Y_avg.reshape((W, W)), cmap=plt.cm.binary, interpolation='nearest')
    plt.savefig('postpred_digit2.pdf')
