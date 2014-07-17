import sys
import os
try:
    import microscopes.cxx.mixture
    assert microscopes.cxx.mixture
    sys.exit(0)
except ImportError:
    pass

from subprocess import check_call
# assumes a git checkout of microscopes-mixturemodel already exists,
# setup by before_install_microscopes_mixturemodel.py
check_call(['make', 'travis_install'], cwd='mixturemodel')
