"""Nucleon structure functions: regression vs the C++ NucleonStructure, WW accuracy."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from taggedpd import GRSV2000Structure, SLACStructure, default_structure
from taggedpd.nucleon import r1998

REF = os.path.join(os.path.dirname(__file__), "reference")


@pytest.fixture(scope="module")
def cppref():
    """physics-code neutron_sf output after the WW fix (commit e9d1e6a)."""
    return np.loadtxt(os.path.join(REF, "neutron_SF.txt.gz"))


@pytest.fixture(scope="module")
def cppref_old():
    """physics-code output with the old 101-point Riemann-sum WW (pre-e9d1e6a)."""
    return np.loadtxt(os.path.join(REF, "neutron_SF_oldWW.txt.gz"))


def test_regression_unpolarized(cppref):
    Q2, x = cppref[:, 0], cppref[:, 1]
    F1, F2 = SLACStructure(R=0.18, target_mass=True).F1F2(x, Q2, "n")
    assert np.allclose(F2, cppref[:, 3], rtol=1e-4, atol=1e-6)
    assert np.allclose(F1, cppref[:, 2], rtol=1e-4, atol=1e-6)
    FT, FL = SLACStructure(R=0.18).FT_FL(x, Q2, "n")
    assert np.allclose(FT, cppref[:, 4], rtol=1e-4, atol=1e-6)
    assert np.allclose(FL, cppref[:, 5], rtol=1e-3, atol=1e-5)


def test_regression_g1_and_ww(cppref):
    """g1 and the accurate WW g1+g2 vs the fixed C++ (both independent quadratures)."""
    Q2, x = cppref[:, 0], cppref[:, 1]
    s = GRSV2000Structure()
    assert np.allclose(s.g1(x, Q2, "n"), cppref[:, 6], rtol=1e-4, atol=1e-6)
    assert np.allclose(s.g1_plus_g2(x, Q2, "n"), cppref[:, 7], rtol=1e-3, atol=1e-5)


def test_old_cpp_ww_emulation(cppref_old):
    """ww="cpp" reproduces the pre-fix C++ Riemann sum (used for the draft's Fig. 12)."""
    Q2, x = cppref_old[:, 0], cppref_old[:, 1]
    s = GRSV2000Structure(ww="cpp")
    assert np.allclose(s.g1_plus_g2(x, Q2, "n"), cppref_old[:, 7], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("nuc", ["n", "p"])
def test_ww_table_matches_quadrature(nuc):
    s = GRSV2000Structure()
    for Q2 in (1.1, 2.0, 7.3, 55.0):
        for x in (2e-3, 0.03, 0.1, 0.27, 0.6):
            pts = [p for p in np.arange(0.1, 1.0, 0.1) if p > x]
            ref = quad(lambda y: s.g1(y, Q2, nuc) / y, x, 1, points=pts, limit=400)[0]
            assert s.g1_plus_g2(x, Q2, nuc) == pytest.approx(ref, rel=2e-3, abs=2e-6)


def test_ww_integral_identity():
    """int_{x0}^1 g2^WW dx = -x0 (g1+g2)^WW(x0)  (follows from swapping the integrals;
    the x0 -> 0 limit is the Burkhardt-Cottingham sum rule)."""
    s = GRSV2000Structure()
    for x0 in (1e-3, 0.02, 0.2):
        x = np.geomspace(x0, 1, 6001)
        lhs = np.trapezoid(s.g2(x, 3.0, "n"), x)
        assert lhs == pytest.approx(-x0 * s.g1_plus_g2(x0, 3.0, "n"), rel=5e-3, abs=2e-6)


def test_r1998_reasonable():
    R = r1998(np.array([0.05, 0.1, 0.3]), np.array([2.0, 5.0, 10.0]))
    assert np.all((R > 0.05) & (R < 0.5))


def test_kinematic_limits():
    s = default_structure()
    assert s.F1F2(1.2, 3.0)[1] == 0.0
    assert s.g1(1.0, 3.0) == 0.0
    assert np.isfinite(s.g1(5e-5, 0.5))   # clamped outside the grid


def test_gridded_wrapper_reproduces_provider():
    from taggedpd import GriddedStructure
    base = default_structure()
    grid = GriddedStructure(base)
    x = np.array([0.01, 0.05, 0.1, 0.3, 0.5]); Q2 = np.array([2.0, 5.0, 10.0, 30.0, 100.0])
    for nuc in ("n", "p"):
        assert np.allclose(grid.F1F2(x, Q2, nuc)[1], base.F1F2(x, Q2, nuc)[1], rtol=2e-2)
        assert np.allclose(grid.g1(x, Q2, nuc), base.g1(x, Q2, nuc), rtol=2e-2, atol=1e-4)
        assert np.allclose(grid.g1_plus_g2(x, Q2, nuc), base.g1_plus_g2(x, Q2, nuc), rtol=2e-2, atol=1e-4)
    assert np.allclose(grid.FT_FL(x, Q2)[1], base.FT_FL(x, Q2)[1], rtol=3e-2)
