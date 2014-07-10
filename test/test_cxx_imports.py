def test_import_gibbs():
    from microscopes.cxx.kernels.gibbs import assign, assign_fixed, assign_resample, hp
    assert assign and assign_fixed and assign_resample and hp

def test_import_slice():
    from microscopes.cxx.kernels.slice import hp
    assert hp

def test_import_bootstrap():
    from microscopes.cxx.kernels.bootstrap import likelihood
    assert likelihood
