"""Contains a parallel runner implementation, with support
for various backends

"""

from microscopes.common import validator
from microscopes.common.rng import rng
import warnings
import logging
import time
import tempfile
import pickle
import multiprocessing as mp

_logger = logging.getLogger(__name__)

try:
    import multyvac
    _has_multyvac = True
except ImportError:
    _has_multyvac = False


def _mp_work(args):
    runner, niters, seed, varg = args
    if varg is not None:
        import multyvac
        import pickle
        import os
        volume, name = varg
        volume = multyvac.volume.get(volume)
        with open(os.path.join(volume.mount_path, name)) as fp:
            view = pickle.load(fp)
        runner._view = view
    prng = rng(seed)
    runner.run(r=prng, niters=niters)
    if varg is not None:
        runner._view = None
    return runner


def _mvac_list_files_in_dir(volume, path):
    ents = volume.ls(path)
    return [x['path'] for x in ents if x['type'] == 'f']


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
            validator.validate_kwargs(kwargs, ('layer', 'core', 'volume',))
            if 'layer' not in kwargs:
                msg = ('multyvac support requires setting up a layer.'
                       'see scripts in bin')
                raise ValueError(msg)
            self._volume = kwargs.get('volume', None)
            if self._volume is None:
                msg = "use of a volume is highly recommended"
                warnings.warn(msg)
            else:
                volume = multyvac.volume.get(self._volume)
                if not volume:
                    raise ValueError(
                        "no such volume: {}".format(self._volume))

            self._layer = kwargs['layer']
            if (not multyvac.config.api_key or
                    not multyvac.config.api_secret_key):
                raise ValueError("multyvac is not auth-ed")
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

            # XXX(stephentu): multyvac post requests are limited in size
            # (don't know what the hard limit is). so to avoid the limits,
            # we explicitly serialize the views to a file

            if not self._volume:
                # no volume provided for uploads
                self._digests = [None for _ in xrange(len(self._runners))]
                return

            # XXX(stephentu): we shouldn't reach in there like this
            views = [r._view for r in self._runners]
            self._digests = []
            digest_cache = {}
            for v in views:
                cache_key = id(v)
                if cache_key in digest_cache:
                    digest = digest_cache[cache_key]
                else:
                    digest = v.digest()
                    digest_cache[cache_key] = digest
                self._digests.append(digest)

            uploaded = set(_mvac_list_files_in_dir(volume, ""))
            _logger.info("starting view uploads")
            start = time.time()
            for view, digest in zip(views, self._digests):
                if digest in uploaded:
                    continue
                _logger.info("uploaded view-{} since not found", digest)
                f = tempfile.NamedTemporaryFile()
                pickle.dump(view, f)
                f.flush()
                volume.put_file(f.name, 'view-{}'.format(digest))
                f.close()
                uploaded.add(digest)
            _logger.info("view upload took {} seconds", (time.time() - start))

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
            has_volume = bool(self._volume)
            zipped = zip(self._runners, self._digests)
            views = []
            for i, (runner, digest) in enumerate(zipped):
                if has_volume:
                    viewarg = (self._volume, 'view-{}'.format(digest))
                    views.append(runner._view)
                    runner._view = None
                else:
                    viewarg = None
                args = (runner, niters, r.next(), viewarg)
                jids.append(
                    multyvac.submit(
                        _mp_work,
                        args,
                        _ignore_module_dependencies=True,
                        _layer=self._layer,
                        _vol=self._volume,
                        _env=dict(self._env),  # submit() mutates the env
                        _core=self._core,
                        _name='kernels-parallel-runner-{}'.format(i)))
            self._runners = [multyvac.get(jid).get_result() for jid in jids]
            if not views:
                return
            for runner, view in zip(self._runners, views):
                runner._view = view
        else:
            assert False, 'should not be reached'

    def get_latents(self):
        return [runner.get_latent() for runner in self._runners]
