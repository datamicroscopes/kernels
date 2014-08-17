"""Contains a parallel runner implementation, with support
for various backends

"""

from microscopes.common import validator
from microscopes.common.rng import rng
import multiprocessing as mp


def _mp_work(args):
    runner, niters, seed = args
    prng = rng(seed)
    runner.run(r=prng, niters=niters)
    return runner


class runner(object):

    def __init__(self, runners, backend='multiprocessing', **kwargs):
        self._runners = runners
        if backend not in ('multiprocessing',):
            raise ValueError("invalid backend: {}".format(backend))
        self._backend = backend
        if backend == 'multiprocessing':
            validator.validate_kwargs(kwargs, ('processes',))
            if 'processes' not in kwargs:
                kwargs['processes'] = mp.cpu_count()
            validator.validate_positive(kwargs['processes'], 'processes')
            self._processes = kwargs['processes']
        else:
            assert False, 'should not be reached'

    def run(self, r, niters=10000):
        """Run each runner for `niters`, using the backend for parallelism

        Parameters
        ----------
        r : rng
        niters : int, optional

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')
        if self._backend == 'multiprocessing':
            pool = mp.Pool(processes=self._processes)
            args = [(runner, niters, r.next()) for runner in self._runners]
            # map_async() + get() allows us to workaround a bug where
            # control-C doesn't kill multiprocessing workers
            self._runners = pool.map_async(_mp_work, args).get(10000000)
            pool.close()
            pool.join()
        else:
            assert False, 'should not be reached'

    def get_latents(self):
        return [runner.get_latent() for runner in self._runners]
