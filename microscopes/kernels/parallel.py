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
    runner, niters, seed, statearg = args
    if statearg is not None:
        import multyvac
        import pickle
        import os
        volume, name = statearg
        volume = multyvac.volume.get(volume)
        with open(os.path.join(volume.mount_path, name)) as fp:
            runner.expensive_state = pickle.load(fp)
    prng = rng(seed)
    runner.run(r=prng, niters=niters)
    if statearg is not None:
        runner.expensive_state = None
    return runner


def _mvac_list_files_in_dir(volume, path):
    ents = volume.ls(path)
    return [x['path'] for x in ents if x['type'] == 'f']


class runner(object):
    """A parallel runner. Note the parallelism is applied across runners
    (currently, each runner is assumed to be single-threaded).

    Parameters
    ----------
    runners : list of runner objects
    backend : string, one of {'multiprocessing', 'multyvac'}
        Indicates the parallelization strategy to be used across
        runners. Note for the 'multiprocessing' backend, the only
        valid kwarg is 'processes'. For the 'multyvac' backend,
        the valid kwargs are 'layer', 'core', and 'volume'.

    processes : int, optional
        For the 'multiprocessing' backend, the number of processes
        in the process pool. Defaults to the number of processes
        on the current machine.

    layer : string
        The multyvac layer which has the datamicroscopes dependencies
        installed.
    core : string
        The type of multyvac core to use. Defaults currently to 'f2' (the most
        expensive, but powerful core type).
    volume : string, optional
        The volume is highly recommended to work around multyvac's limitations
        regarding passing around large objects (e.g. dataviews). The volume
        must be created beforehand. The runner uses the root directory of the
        volume as a cache.

    Notes
    -----
    To use the multyvac backend, you must first authenticate your machine (e.g.
    by running multyvac setup) beforehand.

    """

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
            # XXX(stephentu): assumes you used the setup multyvac scripts we
            # provide
            self._env['PATH'] = '{}:{}'.format(
                '/home/multyvac/miniconda/envs/build/bin', path)
            self._env['CONDA_DEFAULT_ENV'] = 'build'
            # this is needed for multyvacinit.pybootstrap
            self._env['PYTHONPATH'] = '/usr/local/lib/python2.7/dist-packages'

            # XXX(stephentu): multyvac post requests are limited in size
            # (don't know what the hard limit is). so to avoid the limits,
            # we explicitly serialize the expensive state to a file

            if not self._volume:
                # no volume provided for uploads
                self._digests = [None for _ in xrange(len(self._runners))]
                return

            # XXX(stephentu): we shouldn't reach in there like this
            self._digests = []
            digest_cache = {}
            for runner in self._runners:
                cache_key = id(runner.expensive_state)
                if cache_key in digest_cache:
                    digest = digest_cache[cache_key]
                else:
                    digest = runner.expensive_state_digest()
                    digest_cache[cache_key] = digest
                self._digests.append(digest)

            uploaded = set(_mvac_list_files_in_dir(volume, ""))
            _logger.info("starting state uploads")
            start = time.time()
            for runner, digest in zip(self._runners, self._digests):
                if digest in uploaded:
                    continue
                _logger.info("uploaded state-%s since not found", digest)
                f = tempfile.NamedTemporaryFile()
                pickle.dump(runner.expensive_state, f)
                f.flush()
                # XXX(stephentu) this seems to fail for large files
                #volume.put_file(f.name, 'state-{}'.format(digest))
                volume.sync_up(f.name, 'state-{}'.format(digest))
                f.close()
                uploaded.add(digest)
            _logger.info("state upload took %f seconds", (time.time() - start))

        else:
            assert False, 'should not be reached'

    def run(self, r, niters=10000):
        """Run each runner for `niters`, using the backend supplied in the
        constructor for parallelism.

        Parameters
        ----------
        r : rng
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')
        if self._backend == 'multiprocessing':
            pool = mp.Pool(processes=self._processes)
            args = [(runner, niters, r.next(), None)
                    for runner in self._runners]
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
            expensive_states = []
            for i, (runner, digest) in enumerate(zipped):
                if has_volume:
                    statearg = (self._volume, 'state-{}'.format(digest))
                    expensive_states.append(runner.expensive_state)
                    runner.expensive_state = None
                else:
                    statearg = None
                args = (runner, niters, r.next(), statearg)
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
            if not expensive_states:
                return
            for runner, state in zip(self._runners, expensive_states):
                runner.expensive_state = state
        else:
            assert False, 'should not be reached'

    def get_latents(self):
        """Returns a list of the current state of each of the runners.
        """
        return [runner.get_latent() for runner in self._runners]
