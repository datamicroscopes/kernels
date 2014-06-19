# Gaussian conjugate prior, using the same framework as distributions

import numpy as np
import math

from scipy.special import multigammaln
from distributions.dbg.special import gammaln
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin
from microscopes.distributions.gcp_util import sample_niw

NAME = 'NormalInverseWishart'
Value = float

def score_student_t(x, nu, mu, sigma):
    """
    Eq. 313
    """
    d = x.shape[0]
    term1 = gammaln(nu/2 + d/2) - gammaln(nu/2)
    sigmainv = np.linalg.inv(sigma)
    term2 = -0.5*np.log(np.abs(np.linalg.det(sigma))) - d/2*np.log(nu*math.pi)
    diff = x - mu
    term3 = -0.5*(nu+d)*np.log(1 + 1./nu*np.dot(diff, np.dot(sigmainv, diff)))
    return term1 + term2 + term3

class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self._mu0 = None
        self._lam0 = None
        self._psi0 = None
        self._nu0 = None
        self._D = None

    #def plus_group(self, group)

    def dimension(self):
        return self._D

    def load(self, raw):
        self._mu0 = raw['mu']
        self._lam0 = float(raw['lam'])
        self._psi0 = raw['psi']
        assert self._psi0.shape[0] == self._psi0.shape[1]
        self._nu0 = float(raw['nu'])
        self._D = self._psi0.shape[0]

    #def dump(self)
    #def load_protobuf(self, message)
    #def dump_protobuf(self, message)

class Group(GroupIoMixin):
    def __init__(self):
        self._cnts = None
        self._sum_x = None
        self._sum_xxT = None

    def init(self, shared):
        D = shared._D
        self._cnts = 0
        self._sum_x = np.zeros(D)
        self._sum_xxT = np.zeros((D, D))

    def add_value(self, shared, value):
        self._cnts += 1
        self._sum_x += value
        self._sum_xxT += np.outer(value, value)

    def remove_value(self, shared, value):
        self._cnts -= 1
        self._sum_x -= value
        self._sum_xxT -= np.outer(value, value)

    #def merge(self, shared, source)

    def _post_params(self, shared):
        mu0, lam0, psi0, nu0 = shared._mu0, shared._lam0, shared._psi0, shared._nu0
        n, sum_x, sum_xxT = self._cnts, self._sum_x, self._sum_xxT
        xbar = sum_x / n if n else np.zeros(shared._D)
        mu_n = lam0/(lam0 + n)*mu0 + n/(lam0 + n)*xbar
        lam_n = lam0 + n
        nu_n = nu0 + n
        diff = xbar - mu0
        C_n = sum_xxT - np.outer(sum_x, xbar) - np.outer(xbar, sum_x) + np.outer(xbar, xbar)
        psi_n = psi0 + C_n + lam0*n/(lam0+n)*np.outer(diff, diff)
        return mu_n, lam_n, psi_n, nu_n

    def score_value(self, shared, value):
        """
        Eq. 258
        """
        mu_n, lam_n, psi_n, nu_n = self._post_params(shared)
        D = shared._mu0.shape[0]
        Sigma_n = psi_n*(lam_n + 1)/(lam_n*(nu_n-D+1))
        return score_student_t(value, nu_n-D+1, mu_n, Sigma_n)

    def score_data(self, shared):
        """
        Eq. 266
        """
        mu0, lam0, psi0, nu0 = shared._mu0, shared._lam0, shared._psi0, shared._nu0
        mu_n, lam_n, psi_n, nu_n = self._post_params(shared)
        n = self._cnts
        D = shared._mu0.shape[0]
        return multigammaln(nu_n/2, D) + nu0/2*np.linalg.det(psi0) - (n*D/2)*np.log(math.pi) - multigammaln(nu0/2, D) - nu_n/2*np.linalg.det(psi_n) + D/2*np.log(lam0/lam_n)

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    #def load(self, raw)
    #def dump(self)
    #def load_protobuf(self, message)
    #def dump_protobuf(self, message)

class Sampler(object):
    def init(self, shared, group=None):
        mu0, lam0, psi0, nu0 = shared._mu0, shared._lam0, shared._psi0, shared._nu0
        self._mu, self._sigma = sample_niw(mu0, lam0, psi0, nu0)

    def eval(self, shared):
        return np.random.multivariate_normal(self._mu, self._sigma)

#def sample_group(shared, size)

if __name__ == '__main__':
    from microscopes.distributions.gcp_util import random_orthonormal_matrix
    from microscopes.common.dataset import numpy_dataset
    from microscopes.kernels.gibbs import gibbs_assign
    from distributions.dbg.models import bb
    import sys
    gcp = sys.modules['__main__']

    A0 = random_orthonormal_matrix(2)
    psi0 = np.dot(np.dot(A0, np.diag([1.0, 0.1])), A0.T)
    raw = {
        'mu': np.zeros(2),
        'lam': 0.3,
        'psi': psi0,
        'nu' : 3,
    }

    s = Shared()
    s.load(raw)
    g = Group()
    g.init(s)
    g.add_value(s, np.array([1., 2.]))
    g.add_value(s, np.array([-3., 54.]))
    print g.score_value(s, np.array([3., 0.]))
    print g.score_data(s)
    g.remove_value(s, np.array([-3., 54.]))

    from microscopes.models.mixture.dp import DirichletProcess
    dpmm = DirichletProcess(10, {'alpha':2.0}, [gcp, bb], [raw, {'alpha':1.0,'beta':1.0}])
    Y_clustered = dpmm.sample(10)
    Y = np.hstack(Y_clustered)
    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data())
    for _ in xrange(3):
        gibbs_assign(dpmm, dataset.data(shuffle=True))
