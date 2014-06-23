from distributions.dbg.models import bb
from microscopes.common.dataset import numpy_dataset
from microscopes.models.mixture.dp import DirichletProcess
from microscopes.kernels.gibbs import gibbs_assign

from sklearn.datasets import fetch_mldata
mnist_dataset = fetch_mldata('MNIST original')

import numpy as np
import numpy.ma as ma
import matplotlib.pylab as plt
import math

from nose.plugins.attrib import attr

@attr('slow')
def test_mnist():
    Y = mnist_dataset['data'][np.where(mnist_dataset['target'] == 2.)[0]]
    N, D = Y.shape
    W = int(math.sqrt(D))
    assert W * W == D
    dtype = [('', bool)]*D
    N_sampled = 50
    perm = np.random.permutation(np.arange(N))
    inds = perm[:N_sampled]
    Y_sampled = np.array([tuple(y) for y in Y[inds]], dtype=dtype)

    present = D/2
    absent = D-present
    #y_test = tuple(Y[perm[N_sampled]]) # never seen before
    y_test = tuple(Y[perm[0]])
    y_new = ma.masked_array(
        np.array([y_test], dtype=[('', bool)]*D),
        mask=[(False,)*present + (True,)*absent])[0]

    dataset = numpy_dataset(Y_sampled)
    mm = DirichletProcess(N_sampled, {'alpha':2.0}, [bb]*D, [{'alpha':1.0, 'beta':1.0}]*D)
    mm.bootstrap(dataset.data(shuffle=False))

    burnin_niters = 2000
    for _ in xrange(burnin_niters):
        gibbs_assign(mm, dataset.data(shuffle=True))
    print 'finished burnin of', burnin_niters, 'iters'

    def postpred_sample():
        Y_samples = [mm.sample_post_pred(y_new=y_new) for _ in xrange(10000)]
        Y_samples = np.array([list(y) for y in np.hstack(Y_samples)])
        Y_avg = Y_samples.mean(axis=0)
        return Y_avg

    Y_samples = np.array([postpred_sample() for _ in xrange(1)])
    Y_avg = Y_samples.mean(axis=0)

    plt.imshow(Y_avg.reshape((W, W)), cmap=plt.cm.binary)
    plt.savefig('postpred_digit2.pdf')
