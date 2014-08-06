# a simple script to see if our libraries work on multiprocessing

import multiprocessing as mp
import numpy as np

from microscopes.mixture.definition import model_definition
from microscopes.common.rng import rng
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.mixture.model import initialize, deserialize, bind
from microscopes.kernels import gibbs
from microscopes.models import bb

def _make_definition(N, D):
    return model_definition(N, [bb]*D)

def _revive(N, D, data, latent):
    defn = _make_definition(N, D)
    view = numpy_dataview(data)
    latent = deserialize(defn, latent)
    return defn, view, latent

def _work(args):
    (N, D), data, latent = args
    defn, view, latent = _revive(N, D, data, latent)
    r = rng()
    model = bind(latent, view)
    for _ in xrange(1000):
        gibbs.assign(model, r)
    return latent.serialize()

def main():
    N, D = 1000, 10
    data = np.random.random(size=(N, D)) <= 0.8
    data = np.array([tuple(y) for y in data], dtype=[('', bool)]*D)

    procs = mp.cpu_count()
    latents = []
    defn = _make_definition(N, D)
    r = rng()
    for _ in xrange(procs):
        latent = initialize(defn, numpy_dataview(data), r=r)
        latents.append(latent.serialize())

    p = mp.Pool(processes=procs)
    infers = p.map(_work, [((N, D), data, latent) for latent in latents])
    p.close()
    p.join()

    infers = [_revive(N, D, data, infer)[2] for infer in infers]
    for infer in infers:
        print infer.ngroups()

if __name__ == '__main__':
    main()
