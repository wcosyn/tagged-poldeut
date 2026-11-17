"""LF neutron distributions: regression vs the C++ code, sum rules (Table III), probabilistic limits."""

import os

import numpy as np
import pytest

from taggedpd import LFDistributions
from taggedpd.densities import cm_momentum, spectator_lf_from_rest_frame
from taggedpd.sumrules import integrate_distribution

REF = os.path.join(os.path.dirname(__file__), "reference")
SQ2, SQ32 = np.sqrt(2), np.sqrt(1.5)


@pytest.mark.parametrize("wf", ["AV18", "CDBonn", "Paris"])
def test_regression_vs_cpp(wf):
    """Compare with Poldeut::getLFdistributions_new output (6 significant digits,
    C++ wave function linearly interpolated on a 1 MeV grid). Away from nodes the
    agreement is ~1e-5; we allow 2e-3 relative + an absolute floor of 2e-4 of the
    local unpolarized density (values that pass through zero)."""
    d = np.loadtxt(os.path.join(REF, f"lf_dens_{wf}.txt.gz"))
    a, pT = d[:, 0], d[:, 1]
    P = LFDistributions(wf)(a, pT)
    mine = [P.UU, SQ32 * P.TLL_U, SQ2 * P.TLT_U, P.TTT_U / SQ2, P.SL_SL, P.ST_SL, P.SL_ST,
            P.ST_ST_par, P.ST_ST_perp]
    for i in range(d.shape[1] - 2):
        ref = d[:, 2 + i]
        tol = 2e-3 * np.abs(ref) + 2e-4 * P.UU + 1e-6
        bad = np.abs(mine[i] - ref) > tol
        assert not np.any(bad), (i, a[bad][:3], pT[bad][:3], mine[i][bad][:3], ref[bad][:3])


@pytest.fixture(scope="module")
def av18():
    return LFDistributions("AV18")


def test_unpolarized_sum_rules(av18):
    """Eqs. (4.23) and (4.24): number and LF momentum sum rule (= wf norm, exactly).

    AV18 has a slowly decaying tail (4e-5 of the norm sits above k = 3 GeV), hence k_max=30."""
    norm = av18.wf.norm
    kw = dict(k_max=30.0, n_k=3000)
    assert integrate_distribution(av18, "UU", **kw) == pytest.approx(norm, rel=1e-6)
    assert integrate_distribution(av18, "UU", lambda a: 2 - a, **kw) == pytest.approx(norm, rel=1e-6)


def test_tensor_integrals(av18):
    """Eq. (4.25): T_LL integrates to zero with both weights, T_LT with weight 1.

    Notes on the paper's "(same for T_LT, T_TT)":
    * P[T_TT,U] alone does *not* integrate to zero (it is ~ sin^2 theta_k);
    * (2-alpha_p) P[T_LT,U] does not either ((2-alpha_p) = 1 - k cos(theta_k)/E is
      not even in cos(theta_k)).
    Both only drop out of the cross section / sum rules through the
    cos(2phi_p - 2phi_TT), cos(phi_p - phi_TL) factors of Eq. (4.22) under the
    phi_p integration. Checked here explicitly.
    """
    scale = integrate_distribution(av18, "UU")
    for name, w in (("TLL_U", None), ("TLL_U", lambda a: 2 - a), ("TLT_U", None)):
        assert abs(integrate_distribution(av18, name, w)) < 1e-8 * scale
    assert integrate_distribution(av18, "TTT_U") == pytest.approx(-0.222, abs=2e-3)
    assert integrate_distribution(av18, "TLT_U", lambda a: 2 - a) == pytest.approx(0.0214, abs=5e-4)


@pytest.mark.parametrize("name,value,tol", [
    ("SL_SL", 0.911, 1e-3),            # I[SL] (A1), exact column of Table III
    ("ST_ST_par", 1 - 0.125, 1e-3),    # 1 - delta_par (A20)
    ("ST_ST_perp", 1 - 0.0512, 2e-4),  # 1 - delta_perp (A22)
    ("ST_SL", -0.0568, 5e-4),          # eps[SL] (A11)
    ("SL_ST", 0.0550, 2e-4),           # eps[ST] (A28)
])
def test_table_III(av18, name, value, tol):
    assert integrate_distribution(av18, name) == pytest.approx(value, abs=tol)


def test_spin_sum_rules_vs_D_state(av18):
    """(A10)/(A25)/(A27): 1 - 3/2 omega_2 up to O(k^2/m^2) (~3e-3)."""
    target = 1 - 1.5 * av18.wf.norm_D
    half = 0.5 * (integrate_distribution(av18, "ST_ST_par") + integrate_distribution(av18, "ST_ST_perp"))
    assert half == pytest.approx(target, abs=3e-3)
    a27 = 0.5 * (integrate_distribution(av18, "ST_ST_par", lambda a: 1 / (2 - a))
                 + integrate_distribution(av18, "ST_ST_perp", lambda a: 1 / (2 - a)))
    assert a27 == pytest.approx(target, abs=3e-3)
    assert integrate_distribution(av18, "SL_SL") == pytest.approx(target, abs=3e-3)
    # (A24): (delta_par + delta_perp)/2 = 3/2 omega_2 (exact in the NR expansion)
    d_par = 1 - integrate_distribution(av18, "ST_ST_par")
    d_perp = 1 - integrate_distribution(av18, "ST_ST_perp")
    assert 0.5 * (d_par + d_perp) == pytest.approx(1.5 * av18.wf.norm_D, abs=3e-3)


def test_kT_zero_limits(av18):
    """(4.28)/(4.32), (4.51)-(4.53): at pT = 0 the Melosh terms vanish."""
    a = np.linspace(0.45, 1.55, 23)
    P = av18(a, 0.0)
    assert np.allclose(P.ST_SL, 0) and np.allclose(P.SL_ST, 0)
    assert np.allclose(P.ST_ST_par, P.ST_ST_perp)
    assert np.allclose(P.R_LL, 1.0)                          # (4.51)
    assert np.allclose(P.R_TL, 0.5) and np.allclose(P.R_LT, 0.5)  # (4.53)
    E, kz, kT, k = cm_momentum(a, 0.0)
    f0, f2 = av18.wf.f0f2(k)
    R_TT = 1 - 9 / 8 * f2**2 / ((f0 + f2 / (2 * SQ2)) ** 2 + 9 / 8 * f2**2)   # (4.52)
    assert np.allclose(P.R_TT, R_TT, rtol=1e-10)


def test_probabilities_bounded(av18):
    """Pure-state distributions (4.45), (4.47) are positive => ratios in [0, 1]."""
    a, pT = np.meshgrid(np.linspace(0.4, 1.6, 61), np.linspace(0, 0.6, 61))
    P = av18(a, pT)
    for s in (+1, -1):
        for f in (P.P_L_L, P.P_T_L, P.P_T_T, P.P_L_T):
            assert np.all(f(s) >= -1e-12 * P.UU)
    for R in (P.R_LL, P.R_TL, P.R_TT, P.R_LT):
        assert np.all((R > -1e-10) & (R < 1 + 1e-10))


def test_node_semicircle(av18):
    """All polarized distributions vanish where f0 = f2/sqrt2 (k ~ 0.3 GeV)."""
    k = 0.29876   # AV18 node (see test_wavefunctions)
    for th in (0.3, 1.0, 2.0):
        E = np.sqrt(k * k + 0.93891884**2)
        a = 1 + k * np.cos(th) / E
        P = av18(a, k * np.sin(th))
        for v in (P.SL_SL, P.ST_SL, P.ST_ST_par, P.ST_ST_perp, P.SL_ST):
            assert abs(v) < 1e-3 * P.UU


def test_rest_frame_conversion():
    """alpha_p from Eq. (2.13) and back via Eq. (3.20): |k| ~ |p_p| for small momenta."""
    p = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.02], [0.0, 0.1, -0.1]])
    a, pT = spectator_lf_from_rest_frame(p)
    from taggedpd.constants import M_DEUTERON, M_NUCLEON
    assert a[0] == pytest.approx(2 * M_NUCLEON / M_DEUTERON, rel=1e-12)
    assert np.allclose(pT, [0, 0.05, 0.1])
