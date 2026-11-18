"""Tagged structure functions, master formula, asymmetries and the x-integrated sum rules."""

import numpy as np
import pytest

from taggedpd import Kinematics, NucleonStructure, Polarization, TaggedDIS
from taggedpd.constants import M_DEUTERON, TWO_2PI3
from taggedpd.sumrules import _k_grid, alpha_pT_from_k, integrate_distribution


class ToyStructure(NucleonStructure):
    """Analytic toy SFs so that the x integrals in the sum rules are known exactly.

    F2 = x(1-x)^3 (int_0^1 F2 = 1/20), FT = F2/(x(1+R)),
    g1 = (1-x)^3 (int = 1/4), g1+g2 = WW of that, analytically.
    """
    name = "toy"
    R = 0.2

    def F1F2(self, x, Q2, nucleon="n"):
        x = np.asarray(x, float)
        F2 = np.where((x > 0) & (x < 1), x * (1 - x) ** 3, 0.0)
        return F2 / (2 * x * (1 + self.R)), F2

    def g1(self, x, Q2, nucleon="n"):
        x = np.asarray(x, float)
        return np.where((x > 0) & (x < 1), (1 - x) ** 3, 0.0)

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        x = np.asarray(x, float)
        xs = np.clip(x, 1e-300, 1)
        val = -np.log(xs) - 3 * (1 - xs) + 1.5 * (1 - xs**2) - (1 - xs**3) / 3
        return np.where((x > 0) & (x < 1), val, 0.0)


@pytest.fixture(scope="module")
def model():
    return TaggedDIS("AV18")


@pytest.fixture(scope="module")
def toy():
    return TaggedDIS("AV18", structure=ToyStructure())


# ---------------------------------------------------------------------------
# asymmetries vs the closed-form IA expressions of Sec. VI (DIS limit)
# ---------------------------------------------------------------------------
GRID_A = np.array([0.6, 0.7, 0.85, 1.0, 1.0, 1.15, 1.3, 1.45])
GRID_PT = np.array([0.0, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0, 0.45])


def test_tensor_asymmetry_parallel_eq_6_12(model):
    kin = Kinematics.from_epsilon(0.1 * (2 - GRID_A), 2.0, 0.8)
    P = model.lf(GRID_A, GRID_PT)
    AT = model.tensor_asymmetry(kin, GRID_A, GRID_PT, "parallel")
    assert np.allclose(AT, np.sqrt(2 / 3) * P.TLL_U / P.UU, rtol=1e-12, atol=1e-14)
    assert np.all((AT >= -2 - 1e-9) & (AT <= 1 + 1e-9))


def test_table_II_extremes(model):
    """A_par^T = -2 at |alpha_p-1| ~ 0.3, pT=0 and +1 at alpha_p=1, pT ~ 0.3 GeV."""
    kin = Kinematics.from_epsilon(0.1, 2.0, 0.8)
    a = np.linspace(0.6, 0.8, 2001)
    AT = model.tensor_asymmetry(kin, a, 0.0)
    assert AT.min() == pytest.approx(-2.0, abs=1e-6) and abs(a[np.argmin(AT)] - 0.7) < 0.03
    pT = np.linspace(0.2, 0.4, 2001)
    AT = model.tensor_asymmetry(kin, 1.0, pT)
    assert AT.max() == pytest.approx(1.0, abs=1e-6) and abs(pT[np.argmax(AT)] - 0.3) < 0.01


def test_tensor_asymmetry_perp_eq_6_16(model):
    """A_perp^T = -A_par^T/2 up to the power-suppressed T_TT cos2phi_TT term."""
    kin = Kinematics.from_epsilon(0.1 * (2 - GRID_A), 200.0, 0.8)  # large Q2: tiny pT^2/Q^2
    ATpar = model.tensor_asymmetry(kin, GRID_A, GRID_PT, "parallel")
    ATperp = model.tensor_asymmetry(kin, GRID_A, GRID_PT, "perp")
    assert np.allclose(ATperp, -0.5 * ATpar, atol=2e-3)


def test_vector_asymmetry_parallel_eq_6_4(model):
    kin = Kinematics.from_epsilon(0.1 * (2 - GRID_A), 2.0, 0.8)
    sf = model.structure_functions(kin.x, kin.Q2, GRID_A, GRID_PT)
    P, n = sf["P"], sf["nucleon"]
    D = np.sqrt(1 - kin.epsilon**2)       # D_par[SL] in the DIS limit, Eq. (6.3a)
    ratio = P.SL_SL / (P.UU + P.TLL_U / np.sqrt(6)) * 2 * n["g1"] / (n["FT"] + 0.8 * n["FL"])  # (6.5)
    AV = model.vector_asymmetry(kin, GRID_A, GRID_PT, "parallel")
    assert np.allclose(AV, D * ratio, rtol=1e-10)


def test_vector_asymmetry_perp_eq_6_8(model):
    """A_perp^V = D_perp[SL] F_LSL/Sigma + D_perp[ST] F_LST^cosphiS/Sigma (Eq. (6.8)),
    to leading power; here with exact gamma in the polarization axis."""
    a, pT = GRID_A, GRID_PT
    x = 0.1 * (2 - a)
    Q2, y = 10.0, 0.5
    kin = Kinematics.from_y(x, Q2, y)
    sf = model.structure_functions(x, Q2, a, pT)
    g, e = kin.gamma, kin.epsilon
    Dsl = g * y * (1 - y / 2) * np.sqrt(1 - y) / (1 - y + y * y / 2)       # (6.7a), psi_N = 0
    Dst = y * np.sqrt(1 - y) / (1 - y + y * y / 2)                          # (6.7b)
    sig = sf["UU_T"] + e * sf["UU_L"] - (sf["UTLL_T"] + e * sf["UTLL_L"]) / 6
    expect = (Dsl * sf["LSL"] + Dst * sf["LST_cos_phiS"]) / sig
    AV = model.vector_asymmetry(kin, a, pT, "perp", psi_N=0.0)
    assert np.allclose(AV, expect, rtol=0.05, atol=2e-4)


# ---------------------------------------------------------------------------
# master formula
# ---------------------------------------------------------------------------
def test_phi_average(model):
    kin = Kinematics.from_y(0.15, 3.0, 0.6)
    pol = Polarization(SL=0.3, ST=0.7, phi_S=0.4, TLL=0.2, TLT=-0.3, TTT=0.5,
                       phi_TL=1.1, phi_TT=-0.7)
    sf = model.structure_functions(kin.x, kin.Q2, 1.1, 0.25)
    phis = np.linspace(0, 2 * np.pi, 257)[:-1]
    for le in (0.5, -0.5):
        F = model.assemble(sf, kin.epsilon, phis, le, pol)
        Fav = model.assemble(sf, kin.epsilon, 0.0, le, pol, phi_averaged=True)
        assert np.mean(F) == pytest.approx(Fav, rel=1e-12)


def test_pure_states_sum_to_unpolarized(model):
    kin = Kinematics.from_y(0.15, 3.0, 0.6)
    sf = model.structure_functions(kin.x, kin.Q2, 0.9, 0.2)
    phis = np.linspace(0, 2 * np.pi, 13)
    for th, ph in ((0.0, 0.0), (0.7, 0.3), (np.pi / 2, 1.0)):
        tot = sum(model.assemble(sf, kin.epsilon, phis, 0.5, Polarization.pure_state(L, th, ph))
                  for L in (1, 0, -1))
        unpol = model.assemble(sf, kin.epsilon, phis, 0.5, Polarization())
        assert np.allclose(tot, 3 * unpol, rtol=1e-12)


def test_reduced_xsec_and_dsigma_normalization(model):
    """dsigma = flux * sigma_red / [2(2pi)^3] reproduces the C++ calc_dsigma_unpol expression."""
    a, pT, xn, Q2, s = 1.0, 0.1, 0.05, 6.0, 800.0
    x = xn * (2 - a)
    kin = Kinematics.from_s(x, Q2, s, dis_limit=True)
    ds = model.dsigma(kin, a, pT, phi_averaged=True, nb=False)
    sf = model.structure_functions(x, Q2, a, pT)
    n = sf["nucleon"]
    P = sf["P"].UU
    alpha = 0.00729735253
    # C++: strucunpol = 2[2(2pi)^3] 2 P/(2-a) (FT + eps FL); dsigma = strucunpol alpha^2/(4(2pi)^2) y^2/Q^4/(1-eps)
    struc = TWO_2PI3 * 2 * P / (2 - a) * (n["FT"] + kin.epsilon * n["FL"])
    cpp = struc * alpha**2 / (4 * (2 * np.pi) ** 2) * kin.y**2 / Q2**2 / (1 - kin.epsilon)
    assert ds == pytest.approx(cpp, rel=1e-12)


def test_forbidden_region_is_zero(model):
    kin = Kinematics.from_y(1.2, 3.0, 0.5)
    assert model.reduced_cross_section(kin, 0.9, 0.1) == 0.0     # alpha_p > 2 - x


# ---------------------------------------------------------------------------
# x-integrated sum rules (Secs. V B, V D) with the analytic toy structure functions
# ---------------------------------------------------------------------------
def _xint(model, func, n_k=160, n_c=40, n_x=48):
    r"""int_0^2 dx int dGamma_p func(sf, x), with dGamma_p = (2-alpha) d^3k/E /[2(2pi)^3]
    and x = (2-alpha_p) x_n (so dx = (2-alpha_p) dx_n, x_n in (0,1))."""
    k, wk, c, wc = _k_grid(2.5, n_k, n_c)
    a, pT, E = alpha_pT_from_k(k, c)
    u, wu = np.polynomial.legendre.leggauss(n_x)
    xn = 0.5 * (1 + u)
    wx = 0.5 * wu
    A, PT, XN = a[..., None], pT[..., None], xn[None, None, :]
    X = (2 - A) * XN
    sf = model.structure_functions(X, 4.0, A, PT)
    integrand = func(sf, X) * (2 - A) * (2 - A) * 2 * np.pi * (k * k / E)[..., None] / TWO_2PI3
    return float(np.sum((wk * wc)[..., None] * wx[None, None, :] * integrand))


def test_momentum_sum_rule_5_19(toy):
    lhs = _xint(toy, lambda sf, x: x / 2 * (sf["UU_T"] + sf["UU_L"]))
    assert lhs == pytest.approx(toy.lf.wf.norm * 1 / 20, rel=2e-4)


def test_tensor_momentum_sum_rule(toy):
    lhs = _xint(toy, lambda sf, x: x / 2 * (sf["UTLL_T"] + sf["UTLL_L"]))
    assert abs(lhs) < 1e-6


def test_spin_sum_rule_5_29(toy):
    lhs = _xint(toy, lambda sf, x: 0.5 * sf["LSL"])
    I_SL = integrate_distribution(toy.lf, "SL_SL")
    assert lhs == pytest.approx(I_SL * 2 * 0.25, rel=2e-4)
    assert I_SL == pytest.approx(1 - 1.5 * toy.lf.wf.norm_D, abs=3e-3)


def test_burkhardt_cottingham_5_34(toy):
    """int dx/2 dGamma g2D = (1 - 3/2 omega2) int g2n = 0, up to O(k^2/m^2) of (A27)."""
    gam = lambda x: x * M_DEUTERON / 2.0     # gamma = x M_D / Q at Q2 = 4   # noqa: E731
    g2D = _xint(toy, lambda sf, x: 0.5 * (-0.5 * sf["LSL"] - sf["LST_cos_phiS"] / (2 * gam(x))))
    g1_term = _xint(toy, lambda sf, x: 0.5 * (-0.5 * sf["LSL"]))
    assert abs(g2D) < 0.01 * abs(g1_term)


def test_unphysical_kinematics_give_zero(model):
    kin = Kinematics.from_y(np.array([0.1, 0.1]), 3.0, np.array([0.5, 1.3]))
    out = model.dsigma(kin, 1.0, 0.1)
    assert out[0] > 0 and out[1] == 0.0 and np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# proton vs neutron tagging
# ---------------------------------------------------------------------------
def test_tagging_switch_swaps_active_nucleon(model):
    """tagged="n" uses the proton structure functions with the same deuteron LF distributions.

    Every tagged structure function is (deuteron factor) x (one nucleon input), so the
    n-tagged / p-tagged ratio must equal the ratio of the corresponding proton / neutron input.
    """
    n_tag = TaggedDIS("AV18", structure=model.structure, tagged="n")
    assert (model.tagged, model.active) == ("p", "n") and (n_tag.tagged, n_tag.active) == ("n", "p")
    a, pT = GRID_A, GRID_PT
    x, Q2 = 0.1 * (2 - a), 4.0
    sp = model.structure_functions(x, Q2, a, pT)
    sn = n_tag.structure_functions(x, Q2, a, pT)
    # deuteron side identical
    for name in ("UU", "TLL_U", "SL_SL", "ST_ST_par", "SL_ST"):
        assert np.array_equal(getattr(sp["P"], name), getattr(sn["P"], name))
    xn = sp["nucleon"]["xn"]
    st = model.structure
    FTp, FLp = st.FT_FL(xn, Q2, "p")
    FTn, FLn = st.FT_FL(xn, Q2, "n")
    F2p, F2n = st.F1F2(xn, Q2, "p")[1], st.F1F2(xn, Q2, "n")[1]
    g1p, g1n = st.g1(xn, Q2, "p"), st.g1(xn, Q2, "n")
    ggp, ggn = st.g1_plus_g2(xn, Q2, "p"), st.g1_plus_g2(xn, Q2, "n")
    expect = {"UU_T": FTp / FTn, "UTLL_L": FLp / FLn, "UU_cos_phi": F2p / F2n, "UTTT_tw4": F2p / F2n,
              "LSL": g1p / g1n, "LST_cos_phi_m_phiS": g1p / g1n, "LST_cos_phiS": ggp / ggn}
    for key, ratio in expect.items():
        m = np.abs(sp[key]) > 1e-12 * np.abs(sp["UU_T"])
        assert np.allclose(sn[key][m] / sp[key][m], ratio[m], rtol=1e-12), key


def test_tagging_switch_cross_section_and_asymmetry(model):
    n_tag = TaggedDIS("AV18", structure=model.structure, tagged="n")
    kin = Kinematics.from_s(0.1, 6.0, 800.0, dis_limit=True, per_nucleon=True)
    st = model.structure
    FTp, FLp = st.FT_FL(0.1, 6.0, "p")
    FTn, FLn = st.FT_FL(0.1, 6.0, "n")
    ratio = n_tag.dsigma(kin, 1.0, 0.1, phi_averaged=True) / model.dsigma(kin, 1.0, 0.1, phi_averaged=True)
    e = kin.epsilon
    assert ratio == pytest.approx((FTp + e * FLp) / (FTn + e * FLn), rel=1e-12)
    # g1p > 0 > g1n at x_n = 0.1: the double-spin asymmetry flips sign
    assert model.vector_asymmetry(kin, 1.0, 0.1) < 0 < n_tag.vector_asymmetry(kin, 1.0, 0.1)


def test_tagging_switch_rejects_bad_value():
    with pytest.raises(ValueError):
        TaggedDIS("AV18", tagged="proton")
