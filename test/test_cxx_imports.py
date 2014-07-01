def test_import_gibbs():
    from microscopes.cxx.kernels.gibbs import assign, hp
    assert assign and hp

def test_import_slice():
    from microscopes.cxx.kernels.slice import hp
    assert hp

def test_import_bootstrap():
    from microscopes.cxx.kernels.bootstrap import likelihood
    assert likelihood
