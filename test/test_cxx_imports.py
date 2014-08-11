def test_import_gibbs():
    from microscopes.kernels.gibbs import (
        assign,
        assign_fixed,
        assign_resample,
        hp,
    )
    assert assign and assign_fixed and assign_resample and hp


def test_import_slice():
    from microscopes.kernels.slice import hp, theta
    assert hp and theta
