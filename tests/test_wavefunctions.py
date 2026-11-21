"""Deuteron wave function: normalization (arXiv:2603.23700 Eq. (3.23)/(3.25)) and D-wave sign convention."""

import numpy as np
import pytest
from scipy.integrate import quad

from taggedpd import DeuteronWavefunction, available_wavefunctions

WFS = available_wavefunctions()


@pytest.mark.parametrize("name", WFS)
def test_relativistic_norm_equals_exact_closed_form(name):
    """Eq. (3.23), 4pi int dk k^2/E (f0^2+f2^2), vs the closed form sum_ij C_iC_j/(m_i+m_j).

    With the prescription (3.24) f_L = sqrt(E/4pi) U_L the integral is *identical*
    to the nonrelativistic norm, which for Yukawa sums is known analytically, so
    the quadrature must reproduce it to quadrature precision. (The closed form is
    evaluated in exact rational arithmetic; cross-checked against 40-digit
    mpmath for Paris, where float64 cancellation is worst: 1.000127014125197.)
    """
    wf = DeuteronWavefunction(name)
    integrand = lambda k: 4 * np.pi * k * k / wf.energy(k) * (wf.f0(k) ** 2 + wf.f2(k) ** 2)  # noqa: E731
    num = quad(integrand, 0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-12)[0]
    assert num == pytest.approx(wf.norm, rel=1e-7)
    # D-state probability (A7) likewise
    numD = quad(lambda k: 4 * np.pi * k * k / wf.energy(k) * wf.f2(k) ** 2, 0, np.inf,
                limit=500, epsabs=1e-13, epsrel=1e-12)[0]
    assert numD == pytest.approx(wf.norm_D, rel=1e-7)


@pytest.mark.parametrize("name,PD", [("AV18", 0.0578), ("CDBonn", 0.0486), ("Paris", 0.0578)])
def test_norm_close_to_one_and_D_state_probability(name, PD):
    """The parametrizations are normalized to 1 within 2e-4; P_D as published
    (AV18 5.76-5.78%, CD-Bonn 4.85%, Paris 5.77%). arXiv:2603.23700 Table III quotes
    omega_2 = 0.0576 for AV18 (from the interpolated C++ wave function)."""
    wf = DeuteronWavefunction(name)
    assert abs(wf.norm - 1) < 2e-4
    assert wf.norm_D == pytest.approx(PD, abs=3e-4)


@pytest.mark.parametrize("name", WFS)
def test_D_wave_is_positive(name):
    """arXiv:2603.23700's formulas require a positive D wave in momentum space."""
    wf = DeuteronWavefunction(name)
    k = np.linspace(1e-3, 0.8, 400)
    assert np.all(wf.W(k) > 0)
    # small-k behavior W ~ +c k^2
    assert wf.W(1e-3) > 0 and wf.W(2e-3) / wf.W(1e-3) == pytest.approx(4.0, rel=5e-3)
    # and agrees with the C++ GetWp sign (the factor is +1 for the shipped sets)
    assert wf.d_wave_sign == 1.0


def test_S_wave_node_and_D_over_S_node_AV18():
    """f0 = f2/sqrt2 at k ~ 0.30 GeV (arXiv:2603.23700 Sec. IV D / Eq. (6.13), Table II);
    S wave node at k ~ 0.4 GeV."""
    wf = DeuteronWavefunction("AV18")
    k = np.linspace(0.05, 0.6, 20001)
    r = wf.f0(k) - wf.f2(k) / np.sqrt(2)
    k_node = k[np.where(np.diff(np.sign(r)))[0]]
    assert len(k_node) == 1 and k_node[0] == pytest.approx(0.30, abs=0.005)
    kS = k[np.where(np.diff(np.sign(wf.U(k))))[0]]
    assert 0.35 < kS[0] < 0.45


def test_boundary_conditions():
    for name in WFS:
        wf = DeuteronWavefunction(name)
        assert abs(np.sum(wf._c)) < 1e-12
        for p in (-2, 0, 2):
            assert abs(np.sum(wf._d * wf._m ** p)) < 1e-9 * np.sum(np.abs(wf._d * wf._m ** p))


def test_vectorized_shapes():
    wf = DeuteronWavefunction("cdbonn")
    k = np.random.default_rng(1).uniform(0, 1, (3, 4))
    assert wf.U(k).shape == (3, 4) and wf.f2(k).shape == (3, 4)
    assert np.ndim(wf.U(0.1)) == 0
