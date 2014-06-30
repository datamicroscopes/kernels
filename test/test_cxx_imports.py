def test_import_gibbs_assign():
    from microscopes.cxx.kernels.gibbs import assign, hp
    from microscopes.cxx.mixture.model import state
    assert assign and hp and state

def test_import_bootstrap_likelihood():
    from microscopes.cxx.kernels.bootstrap import likelihood
    from microscopes.cxx.mixture.model import state
    assert likelihood and state
