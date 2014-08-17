import multyvac
import argparse

miniconda_url = (
    'http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh'
)

projects = ('common', 'kernels', 'mixturemodel', 'irm',)

setup_sh = """#!/bin/bash
set -e

sudo apt-get update -qq
sudo apt-get install -y python-software-properties wget git make
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8
wget {miniconda_url} -O miniconda.sh
chmod +x ./miniconda.sh
./miniconda.sh -b
export PATH=/home/multyvac/miniconda/bin:$PATH
conda update --yes conda
conda create -n build --yes python=2.7 numpy scipy nose cython pip cmake
source activate build
conda install --yes -c distributions distributions
conda install --yes -c datamicroscopes pymc eigen3
pip install gitpython
export CC=gcc-4.8
export CXX=g++-4.8

PROJECTS="{projects_list}"
for p in $PROJECTS; do
    git clone "https://github.com/datamicroscopes/$p.git"
done

for p in $PROJECTS; do
    (cd "$p" && make release && cd release && make && make install)
    (cd "$p" && pip install .)
done
""".format(miniconda_url=miniconda_url, projects_list=' '.join(projects))

update_sh = """#!/bin/bash
set -e
export PATH=/home/multyvac/miniconda/bin:$PATH
source activate build
export CC=gcc-4.8
export CXX=g++-4.8

PROJECTS="{projects_list}"
for p in $PROJECTS; do
    (cd "$p" && git pull)
done

for p in $PROJECTS; do
    (cd "$p" && make clean)
    (cd "$p" && make release && cd release && make && make install)
    (cd "$p" && pip install . --upgrade)
done
""".format(projects_list=' '.join(projects))


def run_command(job, cmd):
    """
    taken from multyvac/job.py, but has the ssh process
    inherit stdout/stderr from the current process
    """
    from multyvac.util.cygwin import regularize_path
    import subprocess

    if job.wait_for_open_port(22):
        info = job.ports.get('tcp', {}).get('22')
        address = info['address']
        port = info['port']
        cmd = ('{ssh_bin} -o UserKnownHostsFile=/dev/null '
               '-o StrictHostKeyChecking=no -p {port} -i {key_path} '
               ' multyvac@{address} {cmd}'.format(
                   ssh_bin=job.multyvac._ssh_bin,
                   port=port,
                   key_path=regularize_path(
                       job.multyvac.config.path_to_private_key()),
                   address=address,
                   cmd=cmd)
               )
        p = subprocess.Popen(cmd, shell=True)
        return p
    else:
        job.multyvac.job._logger.info('Cannot SSH into finished job')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', required=True)
    parser.add_argument('--action', required=True, choices=['setup', 'update'])
    args = parser.parse_args()

    print args
    if args.action == 'setup':
        multyvac.layer.create(args.layer)
    layer = multyvac.layer.get(args.layer)

    # XXX: don't hardcode multyvac username
    layer.put_contents(setup_sh, "/home/multyvac/setup.sh", 0755)
    layer.put_contents(update_sh, "/home/multyvac/update.sh", 0755)

    job = layer.modify()
    p = run_command(job, "/home/multyvac/{}.sh".format(args.action))
    p.wait()  # wait for the ssh process to finish, *NOT* the job
    job.snapshot()

if __name__ == '__main__':
    main()
