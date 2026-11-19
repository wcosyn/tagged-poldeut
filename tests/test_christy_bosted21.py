"""PRELIMINARY Christy 2021 fit (taggedpd.christy_bosted21, port of SIMC F1F2IN21_v1.0.f).

Reference tables: tests/reference/christy_bosted21_{dp,sp}.txt.gz, produced by
tests/reference/fortran/cb21_reference.f linked against SIMC's F1F2IN21_v1.0.f, see the README
there. Columns: W Q2 | rescsp sigT sigL | rescsn sigT sigL | F1F2IN21 p: F1 F2 | n: F1 F2 | d: F1 F2
"""

import os
import warnings

import numpy as np
import pytest

from taggedpd import christy_bosted21 as c21
from taggedpd.christy_bosted21 import PreliminaryFitWarning
from taggedpd.photoabsorption import ChristyBosted2021, default_photoabsorption, photoproduction

REF = os.path.join(os.path.dirname(__file__), "reference")


def _load(tag):
    return np.loadtxt(os.path.join(REF, f"christy_bosted21_{tag}.txt.gz"))


@pytest.fixture(scope="module")
def tables():
    dp, sp = _load("dp"), _load("sp")
    W, Q2 = dp[:, 0], dp[:, 1]
    w2 = W * W
    cols = list(c21.rescsp(w2, Q2)) + list(c21.rescsn(w2, Q2))
    for t in "pnd":
        cols += list(c21.f1f2in21(t, Q2, w2))
    return dp, sp, np.column_stack(cols)


def test_regression_double_precision_fortran(tables):
    dp, _, py = tables
    ref = dp[:, 2:]
    fin = np.isfinite(ref)
    assert np.allclose(py[fin], ref[fin], rtol=1e-12, atol=1e-300)
    # the only non-finite reference entries: F2 of p and n at Q2 = 0 (0/0 in SF); port gives 0
    q0 = dp[:, 1] == 0
    assert np.all(~fin[:, [5, 7]] == q0[:, None]) and fin[:, [0, 1, 2, 3, 4, 6, 8, 9]].all()
    assert np.all(py[q0][:, [5, 7]] == 0.0)


def test_regression_fortran_as_shipped(tables):
    """vs single-precision-literal build; Q2 = 0.05 excluded: there the shipped build's
    'q2 < 0.05' compares with float32(0.05) > 0.05 and takes the low-Q2 smearing branch."""
    dp, sp, py = tables
    ref = sp[:, 2:]
    keep = (sp[:, 1] != 0.05)
    fin = np.isfinite(ref) & keep[:, None]
    assert np.allclose(py[fin], ref[fin], rtol=2e-5, atol=1e-12)
    # and at Q2 = 0.05 the shipped build equals our low-Q2 branch just below 0.05
    q5 = sp[:, 1] == 0.05
    W = sp[q5, 0]
    low = c21.f1f2in21("d", np.nextafter(0.05, 0), W**2)[0]
    assert np.allclose(low, sp[q5, 10], rtol=2e-4, atol=1e-10)


def test_smearing_normalization():
    """Smearing functions are normalized at gamma = 1 (Q2 -> 0 / Bjorken limit)."""
    y = np.linspace(0.001, 1.999, 20001)
    f11, f12, f2 = c21.getfy(np.ones_like(y), y)
    assert np.trapezoid(f11, y) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(f2, y) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(f12, y) == pytest.approx(0.0, abs=1e-3)


def test_real_photon_limit():
    W = np.linspace(1.1, 4.0, 40)
    sTp, sLp = c21.rescsp(W**2, 0.0)
    sTn, sLn = c21.rescsn(W**2, 0.0)
    assert np.all(sTp > 0) and np.all(sTn > 0) and np.all(sLp == 0) and np.all(sLn == 0)
    for t in "pnd":
        F1, F2 = c21.f1f2in21(t, 0.0, W**2)
        assert np.all(F1 > 0) and np.all(F2 == 0)


def test_deuteron_close_to_free_sum_off_resonance():
    """Smearing only redistributes strength: away from sharp structure d ~ p + n (per nucleus)."""
    W = np.array([2.0, 2.5, 3.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        m = default_photoabsorption(fit="2021")
        d = photoproduction(W, "d", model=m)
        pn = photoproduction(W, "p", model=m) + photoproduction(W, "n", model=m)
    assert np.allclose(d, pn, rtol=0.03)


def test_preliminary_warning_and_labels():
    with pytest.warns(PreliminaryFitWarning, match="PRELIMINARY"):
        m = ChristyBosted2021()
    assert m.preliminary and "PRELIMINARY" in m.name
    with pytest.warns(PreliminaryFitWarning):
        pw = default_photoabsorption(fit="2021")
    assert "PRELIMINARY" in pw.describe()
    with pytest.warns(PreliminaryFitWarning):
        from taggedpd import ChristyBostedStructure
        s = ChristyBostedStructure(version="2021")
    assert "PRELIMINARY" in s.name
    # the 2009 default does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", PreliminaryFitWarning)
        default_photoabsorption()
        photoproduction(1.5, "n")
    with pytest.raises(ValueError):
        default_photoabsorption(fit="2015")


def test_photoproduction_2021_consistency():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        m = default_photoabsorption(fit="2021")
        W = np.linspace(1.1, 4.4, 30)
        assert np.allclose(photoproduction(W, "p", model=m), c21.rescsp(W**2, 0.0)[0], rtol=1e-13)
        assert np.allclose(photoproduction(W, "n", model=m), c21.rescsn(W**2, 0.0)[0], rtol=1e-13)
        F1d = photoproduction(W, "d", "F1", model=m)
        assert np.allclose(F1d, c21.rescsd(W**2, 0.0)[0], rtol=1e-13)
        _, src = photoproduction(np.array([1.0, 1.5, 4.6]), "p", model=m, return_source=True)
        assert list(src) == ["below pion threshold", "Christy 2021 (PRELIMINARY)", ""]


def test_2021_vs_2009_sanity():
    """Both fits describe the same proton photoabsorption data: agree to ~15% in the resonance region."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        m21 = default_photoabsorption(fit="2021")
        W = np.linspace(1.15, 3.0, 50)
        r = photoproduction(W, "p", model=m21) / photoproduction(W, "p")
    assert np.all(np.abs(r - 1) < 0.15)


def test_structure_provider_2021():
    from taggedpd import ChristyBostedStructure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        s = ChristyBostedStructure(version="2021")
    x = np.array([0.2, 0.4, 0.6]); Q2 = np.array([1.0, 2.0, 3.0])
    for nuc in ("p", "n"):
        F1, F2, R = s.F1F2R(x, Q2, nuc)
        w2 = s.W2(x, Q2)
        F1r, F2r = c21.f1f2in21(nuc, Q2, w2)
        assert np.allclose(F1, F1r) and np.allclose(F2, F2r)
        FT, FL = s.FT_FL(x, Q2, nuc)
        # F_L from the fit's sigma_L/sigma_T equals ((1+gamma^2)F2 - 2xF1)/x of SF
        g2 = 1 + 4 * 0.938272**2 * x**2 / Q2
        assert np.allclose(FL, (g2 * F2 - 2 * x * F1) / x, rtol=2e-3)


def test_quantity_conventions_2021():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        m = default_photoabsorption(fit="2021")
    W = np.linspace(1.15, 3.5, 15)
    for Q2 in (0.02, 1.0, 5.0):
        for tgt, (sT, sL) in (("p", c21.rescsp(W**2, Q2)), ("n", c21.rescsn(W**2, Q2))):
            assert np.allclose(photoproduction(W, tgt, "R", Q2=Q2, model=m), sL / sT, rtol=1e-9)
            assert np.allclose(photoproduction(W, tgt, "sigma_L", Q2=Q2, model=m), sL, rtol=1e-9)
            assert np.allclose(photoproduction(W, tgt, "F_L", Q2=Q2, model=m),
                               (sL / sT) * photoproduction(W, tgt, "F_T", Q2=Q2, model=m), rtol=1e-9)
        F1d, F2d, FLd = c21.rescsd(W**2, Q2)
        assert np.allclose(photoproduction(W, "d", "xF_L", Q2=Q2, model=m), FLd, rtol=1e-9, atol=1e-14)
        assert np.allclose(photoproduction(W, "d", "F_T", Q2=Q2, model=m), 2 * F1d, rtol=1e-13)
