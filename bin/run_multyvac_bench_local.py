import multyvac
import argparse

run_sh = """#!/bin/bash
export PATH=/home/multyvac/miniconda/bin:$PATH
source activate build
cd /home/multyvac/kernels/bin
python run_multivac_bench_remote.py --results-dir {results} --benchmark "$@"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', required=True)
    parser.add_argument('--scripts-volume', required=True)
    parser.add_argument('--results-volume', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()

    print args

    scripts_vol = multyvac.volume.get(args.scripts_volume)
    results_vol = multyvac.volume.get(args.results_volume)
    scripts_vol.put_contents(
        run_sh.format(results=results_vol.mount_path),
        "run.sh")

    def runbench(bench):
        jid = multyvac.shell_submit(
            "/bin/bash {}/run.sh {}".format(scripts_vol.mount_path, bench),
            _name='bench1',
            _core='f2',
            _layer=args.layer,
            _vol=[args.scripts_volume, args.results_volume])
        return multyvac.get(jid)

    jobs = [runbench(b) for b in ('mixturemodel', 'irm')]

    if not args.wait:
        return

    for job in jobs:
        job.wait()


if __name__ == '__main__':
    main()
