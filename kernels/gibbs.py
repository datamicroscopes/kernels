import numpy as np
from models.mixture.dp import DPMM
from distributions.dbg.random import sample_discrete_log

class numpy_dataset(object):
    class numpy_iter(object):
        def __init__(self, Y):
            self._Y = Y
            self._i = 0
            self._N = Y.shape[0]
        def __iter__(self):
            return self
        def next(self):
            if self._i == self._N:
                raise StopIteration()
            self._i += 1
            return self._i-1, Y[self._i-1]

    def __init__(self, Y):
        self._Y = Y

    def data(self):
        return numpy_dataset.numpy_iter(self._Y)

    def size(self):
        return self._Y.shape[0]

def repeat_dataset(dataset, n):
    class repeat_iter(object):
        def __init__(self, dataset, n):
            self._dataset = dataset
            self._n = n
            self._i = 0
            self._cur = dataset.data()
        def __iter__(self):
            return self
        def next(self):
            try:
                return self._cur.next()
            except StopIteration:
                self._i += 1
                if self._i == self._n:
                    raise
                self._cur = self._dataset.data()
                return self.next()
    return repeat_iter(dataset, n)

def gibbs_bootstrap(dpmm, dataset):
    N = dataset.size()
    assignments = np.zeros(N, dtype=np.int32)

    # bootstrapping assignment
    for i, yi in dataset.data():
        idmap, scores = dpmm.score_value(yi)
        groupid = idmap[sample_discrete_log(scores)]
        dpmm.add_value(groupid, yi)
        assignments[i] = groupid

    return assignments

def gibbs(dpmm, diter, assignments):
    assignments = assignments.copy()
    for i, yi in diter:
        groupid = assignments[i]
        dpmm.remove_value(groupid, yi)
        idmap, scores = dpmm.score_value(yi)
        groupid = idmap[sample_discrete_log(scores)]
        dpmm.add_value(groupid, yi)
        assignments[i] = groupid
    return assignments

if __name__ == '__main__':
    from distributions.dbg.models import bb
    dpmm = DPMM()
    dpmm.init({'alpha':2.0, 'd':0}, [bb, bb], [{'alpha':0.5, 'beta':0.2}]*2)
    Y = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 0]], dtype=np.int32)
    dataset = numpy_dataset(Y)
    x0 = gibbs_bootstrap(dpmm, dataset)
    gibbs(dpmm, repeat_dataset(dataset, 10), x0)
