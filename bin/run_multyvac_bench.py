import multyvac
import argparse

run_sh = """#!/bin/bash
export PATH=/home/multyvac/miniconda/bin:$PATH
source activate build
python -c 'from microscopes.common import __version__ as v; print v'
python -c 'from microscopes.mixture import __version__ as v; print v'
python -c 'from microscopes.irm import __version__ as v; print v'
python -c 'from microscopes.kernels import __version__ as v; print v'
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', required=True)
    parser.add_argument('--volume', required=True)
    args = parser.parse_args()

    print "layer:", args.layer, 'volume:', args.volume
    vol = multyvac.volume.get(args.volume)
    vol.put_contents(run_sh, "run.sh")

    jid = multyvac.shell_submit(
        "/bin/bash {}/run.sh".format(vol.mount_path),
        _name='bench1',
        _core='f2',
        _layer=args.layer,
        _vol=args.volume)
    job = multyvac.get(jid)
    print job.get_result()

if __name__ == '__main__':
    main()
