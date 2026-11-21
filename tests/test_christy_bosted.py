"""Christy-Bosted port (taggedpd.christy_bosted) and the photoproduction interface.

Reference tables: tests/reference/christy_bosted_{dp,sp}.txt.gz, produced by
tests/reference/fortran/cb_reference.f linked against nucqcd Tools/F1F209_test.f
("sp": as shipped; "dp": compiled with double-precision literals), see the README there.
Columns: W Q2 | christy507 F1 R sigT sigL | resmodd F1avg | f1f2in09 p: F1 F2 R | n: F1 F2 R | d: F1 F2 R
"""

import os

import numpy as np
import pytest

from taggedpd.constants import M_DEUTERON
from taggedpd import christy_bosted as cb
from taggedpd.inclusive import (BelowThreshold, Blend, ChristyBosted, PhotoabsorptionModel,
                                      Piecewise, PION_THRESHOLD, default_photoabsorption,
                                      nu_from_W, W_from_nu, photoproduction)

REF = os.path.join(os.path.dirname(__file__), "reference")


def _load(tag):
    return np.loadtxt(os.path.join(REF, f"christy_bosted_{tag}.txt.gz"))


def _python_table(W, Q2):
    w2 = W * W
    above = w2 > cb.W2_MIN
    w2s = np.where(above, w2, 2.0)
    F1, R, sT, sL = cb.christy507(w2s, Q2)
    cols = [np.where(above, v, 0.0) for v in (F1, R, sT, sL, cb.resmodd(w2s, Q2))]
    for t in "pnd":
        cols.extend(cb.f1f2in09(t, Q2, w2))   # binned=True: FORTRAN behaviour
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def ref_dp():
    return _load("dp")


@pytest.fixture(scope="module")
def ref_sp():
    return _load("sp")


def test_regression_double_precision_fortran(ref_dp):
    """Port == FORTRAN with double-precision literals, to rounding (all 14 columns)."""
    py = _python_table(ref_dp[:, 0], ref_dp[:, 1])
    assert np.allclose(py, ref_dp[:, 2:], rtol=1e-12, atol=1e-300)


def test_regression_fortran_as_shipped(ref_sp):
    """Port vs the FORTRAN as linked by nucqcd (single-precision literals)."""
    py = _python_table(ref_sp[:, 0], ref_sp[:, 1])
    ref = ref_sp[:, 2:]
    pn = slice(0, 11)          # proton, average, p and n columns
    assert np.allclose(py[:, pn], ref[:, pn], rtol=3e-5, atol=1e-12)
    assert np.allclose(py[:, 11:], ref[:, 11:], rtol=2e-3, atol=1e-12)   # deuteron: bin edges shift


def test_real_photon_limit():
    W = np.linspace(1.1, 3.0, 50)
    F1, R, sT, sL = cb.christy507(W**2, 0.0)
    assert np.all(sT > 0) and np.all(sL == 0) and np.all(R == 0)
    for t in "pnd":
        F1t, F2t, Rt = cb.f1f2in09(t, 0.0, W**2)
        assert np.all(F1t > 0) and np.all(F2t == 0) and np.all(Rt == 0)
    # sigma_T <-> F1 conversion is the one built into the fit
    assert np.allclose(cb.sigmaT_from_F1(F1, W**2), sT, rtol=1e-12)
    assert np.allclose(cb.F1_from_sigmaT(sT, W**2), F1, rtol=1e-12)


def test_small_Q2_continuity():
    """Q2 -> 0 is continuous (the fit has fast structure below 1e-3 GeV^2, but no jump)."""
    W2 = np.linspace(1.2, 9.0, 40)
    for t in "pnd":
        a = cb.f1f2in09(t, 0.0, W2)[0]
        b = cb.f1f2in09(t, 1e-9, W2)[0]
        assert np.allclose(a, b, rtol=1e-5)


def test_neutron_is_twice_average_minus_proton():
    W2 = np.linspace(1.2, 9.0, 30)
    for Q2 in (0.0, 0.7, 4.0):
        for binned in (True, False):
            n = cb.f1f2in09("n", Q2, W2, binned=binned)[0]
            p = cb.f1f2in09("p", Q2, W2)[0]
            avg = cb.resmodd(W2, Q2, binned=binned)
            assert np.allclose(n, 2 * avg - p, rtol=1e-13)


def test_binned_vs_smooth():
    """The FORTRAN's 1 MeV binning only matters at the percent level above W = 1.2 GeV."""
    W = np.linspace(1.2, 3.0, 3000)
    for Q2 in (0.0, 1.0):
        a = cb.resmodd(W**2, Q2, binned=True)
        b = cb.resmodd(W**2, Q2, binned=False)
        assert np.max(np.abs(a / b - 1)) < 1.5e-2
        d1 = cb.f1f2in09("d", Q2, W**2, binned=True)[0]
        d2 = cb.f1f2in09("d", Q2, W**2, binned=False)[0]
        assert np.max(np.abs(d1 / d2 - 1)) < 5e-3


def test_resmodd_range():
    assert cb.resmodd(1.0, 0.5) == 0.0 and cb.resmodd(26.0, 0.5) == 0.0 and cb.resmodd(2.0, 12.0) == 0.0
    assert cb.f1f2in09("p", 0.5, 1.1)[0] == 0.0          # below the pion threshold


def test_broadcasting_and_shapes():
    W2 = np.linspace(1.3, 4.0, 6).reshape(2, 3)
    for t in "pnd":
        for arr in cb.f1f2in09(t, np.array([[0.0], [1.0]]), W2):
            assert arr.shape == (2, 3)
    assert np.ndim(cb.f1f2in09("d", 0.5, 2.0)[0]) == 0


# ---------------------------------------------------------------------------
# photoproduction interface
# ---------------------------------------------------------------------------
def test_default_model_pieces():
    W = np.array([1.0, 1.08, 1.5, 3.05, 3.5])
    sig, src = photoproduction(W, "p", return_source=True)
    assert sig[0] == 0.0 and src[0] == "below pion threshold"
    assert src[1] == "Christy-Bosted (threshold extrapolation)"
    assert src[2] == "Christy-Bosted" and src[3] == "Christy-Bosted"
    assert np.isnan(sig[4]) and src[4] == ""
    sig_n, src_n = photoproduction(W, "n", return_source=True)
    assert src_n[3] == "" and np.isnan(sig_n[3])          # neutron fit stops at 3.0
    _, src_d = photoproduction(np.array([1.1, 1.5]), "d", return_source=True)
    assert list(src_d) == ["Christy-Bosted (threshold extrapolation)", "Christy-Bosted"]


def test_photoproduction_matches_fit():
    W = np.linspace(1.1, 3.0, 40)
    for t in "pnd":
        # deuteron: Part II convention F1^D = M_D W1, the fit's native F1 is M W1 (M = PM)
        fac = M_DEUTERON / cb._PM if t == "d" else 1.0
        F1 = cb.f1f2in09(t, 0.0, W**2, binned=False)[0]
        assert np.allclose(photoproduction(W, t, "F1"), fac * F1, rtol=1e-13)
        assert np.allclose(photoproduction(W, t), cb.sigmaT_from_F1(F1, W**2), rtol=1e-13)
    # proton sigma_T is directly the fit's sigma_T
    assert np.allclose(photoproduction(W, "p"), cb.christy507(W**2, 0.0)[2], rtol=1e-12)
    # Q2 dependence passes through
    assert np.allclose(photoproduction(W, "p", "F1", Q2=2.0), cb.christy507(W**2, 2.0)[0], rtol=1e-12)


def test_outside_policies():
    m = default_photoabsorption(outside="zero")
    assert photoproduction(4.0, "p", model=m) == 0.0
    m = default_photoabsorption(outside="raise")
    with pytest.raises(ValueError, match="no photoabsorption model"):
        photoproduction(np.array([1.5, 4.0]), "p", model=m)
    with pytest.raises(ValueError):
        photoproduction(1.5, "x")
    with pytest.raises(ValueError):
        photoproduction(1.5, "p", quantity="F_LT")


class _ToyRegge(PhotoabsorptionModel):
    name = "toy Regge"
    targets = ("p", "n")

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= 2.0) & (Q2 == 0)

    def sigma_T(self, W, Q2, target):
        s = np.asarray(W, float) ** 2
        return 100.0 * s**0.08 + 50.0 * s**-0.45


def test_custom_model_precedence_and_conversion():
    toy = _ToyRegge()
    model = Piecewise([toy, *default_photoabsorption().models])
    W = np.array([1.5, 2.5, 4.0])
    sig, src = photoproduction(W, "p", model=model, return_source=True)
    assert list(src) == ["Christy-Bosted", "toy Regge", "toy Regge"]
    assert sig[1] == pytest.approx(toy.sigma_T(2.5, 0.0, "p"))
    # F1 follows from sigma_T through Hand's convention
    assert photoproduction(2.5, "p", "F1", model=model) == pytest.approx(
        cb.F1_from_sigmaT(toy.sigma_T(2.5, 0.0, "p"), 2.5**2))
    # the deuteron is not described by the toy -> falls through to Christy-Bosted
    _, src_d = photoproduction(np.array([2.5]), "d", model=model, return_source=True)
    assert src_d[0] == "Christy-Bosted"
    # a single (non-piecewise) model returns NaN outside its domain
    assert np.isnan(photoproduction(1.5, "p", model=toy))


def test_blend_is_continuous():
    cbm, toy = ChristyBosted(), _ToyRegge()
    blend = Blend(cbm, toy, 2.0, 2.5)
    model = Piecewise([blend, toy, cbm])
    eps = 1e-9
    for Wb in (2.0, 2.5):
        lo = photoproduction(Wb - eps, "p", model=model)
        hi = photoproduction(Wb + eps, "p", model=model)
        assert lo == pytest.approx(hi, rel=1e-6)


def test_threshold_and_kinematics_helpers():
    assert PION_THRESHOLD == pytest.approx(0.9382727 + 0.135)
    assert BelowThreshold().sigma_T(np.array([1.0, 1.05]), 0.0, "n").tolist() == [0.0, 0.0]
    nu = np.array([0.2, 0.5, 2.0])
    assert np.allclose(nu_from_W(W_from_nu(nu)), nu)
    assert np.allclose(nu_from_W(W_from_nu(nu, Q2=0.3), Q2=0.3), nu)


def test_delta_peak_magnitude():
    """Sanity: the gamma p Delta(1232) peak is ~0.5 mb (PDG total photoabsorption)."""
    W = np.linspace(1.15, 1.3, 400)
    sig = photoproduction(W, "p")
    assert 400 < sig.max() < 600 and 1.20 < W[np.argmax(sig)] < 1.24


# ---------------------------------------------------------------------------
# NucleonStructure provider
# ---------------------------------------------------------------------------
def test_structure_provider():
    from taggedpd import ChristyBostedStructure, CombinedStructure, GRSV2000Structure
    s = ChristyBostedStructure()
    x = np.array([0.2, 0.4, 0.6]); Q2 = np.array([1.0, 2.0, 3.0])
    for nuc in ("p", "n"):
        F1, F2 = s.F1F2(x, Q2, nuc)
        w2 = s.W(x, Q2, nuc) ** 2   # physical nucleon mass in x -> W
        F1r, F2r, R = cb.f1f2in09(nuc, Q2, w2, binned=False)
        assert np.allclose(F1, F1r) and np.allclose(F2, F2r)
        FT, FL = s.FT_FL(x, Q2, nuc)
        assert np.allclose(FL, R * FT, rtol=1e-13)
    assert s.F1F2(1.2, 2.0, "p") == (0.0, 0.0)
    assert s.in_range(0.4, 2.0, "p") and not s.in_range(0.01, 2.0, "p")
    from taggedpd import Kinematics, TaggedDIS
    model = TaggedDIS("AV18", structure=CombinedStructure(s, GRSV2000Structure()))
    sf = model.structure_functions(0.4, 2.0, 1.0, 0.1)
    numeric = [v for v in sf.values() if isinstance(v, (float, np.floating, np.ndarray))]
    assert numeric and all(np.all(np.isfinite(v)) for v in numeric)
    kin = Kinematics.from_y(0.4, 2.0, 0.5)
    assert np.isfinite(model.reduced_cross_section(kin, 1.0, 0.1, 0.0))


# ---------------------------------------------------------------------------
# structure-function conventions (quantity=)
# ---------------------------------------------------------------------------
def _kin(W, Q2, M=cb._PM, MT=None):
    """x = Q^2/(2 M_T nu) (M_T = M for p, n; M_D for d, i.e. x_D), gamma^2 = Q^2/nu^2."""
    nu = (W**2 - M**2 + Q2) / (2 * M)
    return Q2 / (2 * (MT or M) * nu), Q2 / nu**2


@pytest.mark.parametrize("tgt", ["p", "n", "d"])
def test_quantity_conventions_2009(tgt):
    W = np.linspace(1.15, 2.9, 25)
    for Q2 in (0.3, 1.0, 4.0):
        F1, F2, R = cb.f1f2in09(tgt, Q2, W**2, binned=False)
        q = {k: photoproduction(W, tgt, k, Q2=Q2) for k in
             ("sigma_T", "sigma_L", "R", "F1", "F2", "F_T", "F_L", "F_L_scaling", "xF_L")}
        x, g2 = _kin(W, Q2)
        xF_L_native = (1 + g2) * F2 - 2 * x * F1          # invariant: same in both conventions
        if tgt == "d":                                     # Part II: F1^D = M_D W1, x_D
            F1 = F1 * M_DEUTERON / cb._PM
            x, g2 = _kin(W, Q2, MT=M_DEUTERON)
            # Callan-Gross-type check of the convention: x_D F1^D = x_N F1_native
            assert np.allclose(x * q["F1"], _kin(W, Q2)[0] * cb.f1f2in09(tgt, Q2, W**2, binned=False)[0], rtol=1e-12)
        assert np.allclose(q["xF_L"], xF_L_native, rtol=1e-10, atol=1e-14)
        assert np.allclose(q["F1"], F1, rtol=1e-13) and np.allclose(q["F2"], F2, rtol=1e-13)
        assert np.allclose(q["F_T"], 2 * F1, rtol=1e-13)
        # F_L = (1+g^2)F2/x - 2F1 = R F_T, with the fit's own R
        assert np.allclose(q["F_L"], (1 + g2) * F2 / x - 2 * F1, rtol=1e-10, atol=1e-14)
        assert np.allclose(q["F_L"], R * q["F_T"], rtol=1e-10, atol=1e-14)
        assert np.allclose(q["R"], R, rtol=1e-10, atol=1e-14)
        assert np.allclose(q["sigma_L"], R * q["sigma_T"], rtol=1e-10, atol=1e-12)
        # scaling-limit relation and x-weighted F_L
        assert np.allclose(q["F_L_scaling"], F2 / x - 2 * F1, rtol=1e-10, atol=1e-14)
        assert np.allclose(q["F_L_scaling"], q["F_T"] * (R - g2) / (1 + g2), rtol=1e-9, atol=1e-13)
        assert np.allclose(q["xF_L"], x * q["F_L"], rtol=1e-10, atol=1e-14)
        # (1 + gamma^2) F2 = x (F_T + F_L)
        assert np.allclose((1 + g2) * F2, x * (q["F_T"] + q["F_L"]), rtol=1e-10)
    if tgt == "p":
        assert np.allclose(photoproduction(W, "p", "sigma_L", Q2=1.0), cb.christy507(W**2, 1.0)[3], rtol=1e-10)


def test_scaling_relation_breaks_down_at_low_Q2():
    """In the resonance region gamma^2 = Q^2/nu^2 is not small: F_L_scaling < 0 at the Delta, Q^2 = 1."""
    assert photoproduction(1.232, "p", "F_L_scaling", Q2=1.0) < 0 < photoproduction(1.232, "p", "F_L", Q2=1.0)
    # but the two agree when gamma^2 << 1 (large W at fixed Q^2)
    a, b = photoproduction(2.9, "p", "F_L", Q2=0.3), photoproduction(2.9, "p", "F_L_scaling", Q2=0.3)
    x, g2 = _kin(2.9, 0.3)
    assert g2 < 0.02 and b == pytest.approx(a, rel=0.2)


def test_quantities_at_real_photon_point():
    W = np.linspace(1.1, 3.0, 20)
    for tgt in "pnd":
        for k in ("sigma_L", "R", "F2", "F_L", "F_L_scaling", "xF_L"):
            v = photoproduction(W, tgt, k)
            assert np.all(v == 0.0), (tgt, k)
        assert np.allclose(photoproduction(W, tgt, "F_T"), 2 * photoproduction(W, tgt, "F1"))


def test_quantities_for_sigmaT_only_model():
    toy = _ToyRegge()
    model = Piecewise([toy])
    W = np.array([2.5, 3.0])
    assert np.allclose(photoproduction(W, "p", "F_T", model=model),
                       2 * cb.F1_from_sigmaT(toy.sigma_T(W, 0.0, "p"), W**2))
    assert np.all(photoproduction(W, "p", "F_L", model=model) == 0.0)        # Q2 = 0: known
    with pytest.raises(ValueError, match="quantity"):
        photoproduction(W, "p", "FL", model=model)
    blend = Blend(ChristyBosted(), toy, 2.0, 2.5)
    assert blend.evaluate("F_T", 2.25, 0.0, "p") == pytest.approx(
        0.5 * (photoproduction(2.25, "p", "F_T") + 2 * toy.F1(2.25, 0.0, "p")))


def test_sigmaT_only_model_raises_for_F2_at_finite_Q2():
    class Toy(_ToyRegge):
        def domain(self, W, Q2, target):
            W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
            return W >= 2.0
    with pytest.raises(NotImplementedError, match="only sigma_T"):
        photoproduction(2.5, "p", "F_L", Q2=0.5, model=Piecewise([Toy()]))
