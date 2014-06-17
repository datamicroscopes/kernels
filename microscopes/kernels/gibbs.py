import numpy as np
from microscopes.models.mixture.dp import DPMM
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

def gibbs(dpmm, it):
    empty_gids = list(dpmm.empty_groups())
    empty_gid = empty_gids[0] if len(empty_gids) else dpmm.create_group()
    for ei, yi in it:
        gid = dpmm.remove_entity_from_group(ei, yi)
        if not dpmm.nentities_in_group(gid):
            dpmm.delete_group(gid)
        idmap, scores = dpmm.score_value(yi)
        gid = idmap[sample_discrete_log(scores)]
        dpmm.add_entity_to_group(gid, ei, yi)
        if gid == empty_gid:
            empty_gid = dpmm.create_group()

if __name__ == '__main__':
    from distributions.dbg.models import bb
    Y = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 0]], dtype=np.int32)
    N, _ = Y.shape
    dpmm = DPMM(N, {'alpha':2.0}, [bb, bb], [{'alpha':0.5, 'beta':0.2}]*2)
    dataset = numpy_dataset(Y)
    dpmm.bootstrap(dataset.data())
    gibbs(dpmm, repeat_dataset(dataset, 10))
