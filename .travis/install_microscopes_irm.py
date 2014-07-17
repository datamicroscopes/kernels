import sys
import os
try:
    import microscopes.cxx.irm
    assert microscopes.cxx.irm
    sys.exit(0)
except ImportError:
    pass

from subprocess import check_call
# assumes a git checkout of microscopes-irm already exists,
# setup by before_install_microscopes_irm.py
check_call(['make', 'travis_install'], cwd='irm')
