"""LHAPDFStructure: needs the lhapdf python module and the CT18LO and NNPDFpol11_100 sets.

Install the sets with ``lhapdf install CT18LO NNPDFpol11_100``; the tests skip otherwise.
"""

import numpy as np
import pytest
from scipy.integrate import quad

lhapdf = pytest.importorskip("lhapdf")

from taggedpd import (AsNucleonStructure, CombinedStructure, GriddedStructure,  # noqa: E402
                      GRSV2000Structure, Kinematics, LHAPDFStructure, SLACStructure, TaggedDIS,
                      inclusive_sf, resonance_dis_model)
from taggedpd.constants import M_PROTON  # noqa: E402

UNPOL, POL = "CT18LO", "NNPDFpol11_100"
lhapdf.setVerbosity(0)
_missing = {UNPOL, POL} - set(lhapdf.availablePDFSets())
pytestmark = pytest.mark.skipif(bool(_missing), reason=f"LHAPDF sets not installed: {_missing}")

X = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.7])
Q2 = 10.0


@pytest.fixture(scope="module")
def lh():
    return LHAPDFStructure(UNPOL, POL)


def _manual(pdf, x, q2, proton, nf):
    """sum e_q^2 x (q + qbar) straight from LHAPDF, point by point."""
    e2 = {1: 1 / 9, 2: 4 / 9, 3: 1 / 9, 4: 4 / 9, 5: 1 / 9}
    xf = pdf.xfxQ2(x, q2)   # dict over all flavours
    out = 0.0
    for f in range(1, nf + 1):
        fl = f if proton or f > 2 else 3 - f
        out += e2[f] * (xf[fl] + xf[-fl])
    return out


def test_parton_model_sums_and_isospin(lh):
    """F2 and g1 equal the LO parton-model sums; the neutron has u <-> d."""
    u, p = lhapdf.mkPDF(UNPOL, 0), lhapdf.mkPDF(POL, 0)
    for nuc, proton in (("p", True), ("n", False)):
        F1, F2 = lh.F1F2(X, Q2, nuc)
        g1 = lh.g1(X, Q2, nuc)
        for i, x in enumerate(X):
            assert F2[i] == pytest.approx(_manual(u, x, Q2, proton, 5), rel=1e-12)
            assert g1[i] == pytest.approx(0.5 * _manual(p, x, Q2, proton, 3) / x, rel=1e-12)
        assert np.allclose(F1, F2 / (2 * X * 1.18), rtol=1e-12)


def test_shapes_and_edges(lh):
    """Scalars give scalars, x and Q2 broadcast, and everything vanishes outside 0 < x < 1."""
    assert np.ndim(lh.F1F2(0.1, Q2, "p")[1]) == 0
    assert np.ndim(lh.g1(0.1, Q2, "p")) == 0
    q2s = np.array([2.0, 10.0, 100.0])
    assert lh.F1F2(0.1, q2s, "p")[1].shape == (3,)
    assert lh.g1(0.1, q2s, "n").shape == (3,)
    assert lh.F1F2(X[:, None], q2s[None, :], "p")[1].shape == (len(X), 3)
    assert np.allclose(lh.F1F2(0.1, q2s, "p")[1], [lh.F1F2(0.1, q, "p")[1] for q in q2s])
    edge = np.array([0.0, 1.0, 1.2])
    assert np.all(lh.F1F2(edge, Q2, "p")[1] == 0) and np.all(lh.F1F2(edge, Q2, "p")[0] == 0)
    assert np.all(lh.g1(edge, Q2, "p") == 0) and np.all(lh.g1_plus_g2(edge, Q2, "p") == 0)


def test_one_set_only():
    unpol = LHAPDFStructure(UNPOL)
    assert np.isfinite(unpol.F1F2(0.1, Q2, "p")[1])
    with pytest.raises(NotImplementedError):
        unpol.g1(0.1, Q2, "p")
    with pytest.raises(NotImplementedError):
        LHAPDFStructure(polarized_set=POL).F1F2(0.1, Q2, "p")


@pytest.mark.parametrize("nuc", ["p", "n"])
@pytest.mark.parametrize("x", [0.02, 0.1, 0.3, 0.6])
def test_wandzura_wilczek(lh, nuc, x):
    """Tabulated g1 + g2 = int_x^1 dy/y g1(y)."""
    ref = quad(lambda y: lh.g1(y, Q2, nuc) / y, x, 1.0, limit=200, epsabs=1e-7)[0]
    assert lh.g1_plus_g2(x, Q2, nuc) == pytest.approx(ref, rel=2e-3, abs=2e-5)


def test_gridded_matches_direct(lh):
    grid = GriddedStructure(lh)
    x = X[1:-1]
    for nuc in "pn":
        assert np.allclose(grid.F1F2(x, Q2, nuc)[1], lh.F1F2(x, Q2, nuc)[1], rtol=2e-3)
        assert np.allclose(grid.g1(x, Q2, nuc), lh.g1(x, Q2, nuc), rtol=2e-3, atol=1e-4)
        assert np.allclose(grid.g1_plus_g2(x, Q2, nuc), lh.g1_plus_g2(x, Q2, nuc),
                           rtol=2e-3, atol=1e-4)


def test_close_to_default_inputs(lh):
    """Rough sanity check against SLAC F2 and GRSV2000 g1 of the proton, 0.1 <= x <= 0.5.

    Different fits, orders and higher-twist content: at Q2 = 10 the differences are up to
    ~14% (F2 at x = 0.5, where SLAC has target-mass and higher-twist effects LO PDFs lack).
    """
    x = np.array([0.1, 0.2, 0.3, 0.5])
    assert np.allclose(lh.F1F2(x, Q2, "p")[1], SLACStructure().F1F2(x, Q2, "p")[1], rtol=0.2)
    assert np.allclose(lh.g1(x, Q2, "p"), GRSV2000Structure().g1(x, Q2, "p"), rtol=0.2)
    # first moment of g1p at Q2 = 10 (measured ~0.13; this set gives 0.142)
    gamma1p = quad(lambda y: lh.g1(y, Q2, "p"), 1e-4, 1.0, limit=200)[0]
    assert 0.09 < gamma1p < 0.17


def test_in_inclusive_and_tagged(lh):
    """As the DIS part of the inclusive model, and as input to TaggedDIS."""
    m = resonance_dis_model(dis=lh)
    assert inclusive_sf(5.0, "p", "F2", Q2=Q2, model=m) == pytest.approx(
        float(lh.F1F2(Q2 / (5.0**2 - M_PROTON**2 + Q2), Q2, "p")[1]), rel=1e-3)
    td = TaggedDIS("AV18", structure=CombinedStructure(AsNucleonStructure(m), lh))
    kin = Kinematics.from_y(np.array([0.05, 0.3]), 4.0, 0.6)
    assert np.all(np.isfinite(td.reduced_cross_section(kin, 1.0, 0.1, 0.3)))
