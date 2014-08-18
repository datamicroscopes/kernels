"""Contains a parallel runner implementation, with support
for various backends

"""

from microscopes.common import validator
from microscopes.common.rng import rng
import multiprocessing as mp

try:
    import multyvac
    _has_multyvac = True
except ImportError:
    _has_multyvac = False


def _mp_work(args):
    runner, niters, seed = args
    prng = rng(seed)
    runner.run(r=prng, niters=niters)
    return runner


class runner(object):

    def __init__(self, runners, backend='multiprocessing', **kwargs):
        self._runners = runners
        if backend not in ('multiprocessing', 'multyvac',):
            raise ValueError("invalid backend: {}".format(backend))
        self._backend = backend
        if backend == 'multiprocessing':
            validator.validate_kwargs(kwargs, ('processes',))
            if 'processes' not in kwargs:
                kwargs['processes'] = mp.cpu_count()
            validator.validate_positive(kwargs['processes'], 'processes')
            self._processes = kwargs['processes']
        elif backend == 'multyvac':
            if not _has_multyvac:
                raise ValueError("multyvac module not installed on machine")
            validator.validate_kwargs(kwargs, ('layer', 'core',))
            if 'layer' not in kwargs:
                msg = ('multyvac support requires setting up a layer.'
                       'see scripts in bin')
                raise ValueError(msg)
            self._layer = kwargs['layer']
            if (not multyvac.config.api_key or
                    not multyvac.config.api_secret_key):
                raise ValueError("multyvac is not authed")
            # XXX(stephentu): currently defaults to the good stuff
            self._core = kwargs.get('core', 'f2')
            self._env = {}
            jid = multyvac.shell_submit('echo $PATH', _layer=self._layer)
            path = multyvac.get(jid).get_result()
            # XXX(stephentu): fragile, and assumes you used the setup
            # multyvac scripts we provide
            self._env['PATH'] = '{}:{}'.format(
                '/home/multyvac/miniconda/envs/build/bin', path)
            self._env['CONDA_DEFAULT_ENV'] = 'build'
            # this is needed for multyvacinit.pybootstrap
            self._env['PYTHONPATH'] = '/usr/local/lib/python2.7/dist-packages'
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
        elif self._backend == 'multyvac':
            # XXX(stephentu): the only parallelism strategy thus far is every
            # runner gets a dedicated core (multicore=1) on a machine
            jids = []
            for i, runner in enumerate(self._runners):
                args = (runner, niters, r.next())
                jids.append(
                    multyvac.submit(
                        _mp_work,
                        args,
                        _layer=self._layer,
                        _env=self._env,
                        _core=self._core,
                        _name='kernels-parallel-runner-{}'.format(i)))
            self._runners = [multyvac.get(jid).get_result() for jid in jids]
        else:
            assert False, 'should not be reached'

    def get_latents(self):
        return [runner.get_latent() for runner in self._runners]
