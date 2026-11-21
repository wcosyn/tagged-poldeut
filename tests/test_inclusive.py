"""The (W, Q^2) inclusive layer and its adapters to/from the (x, Q^2) NucleonStructure interface."""

import warnings

import numpy as np
import pytest

import taggedpd
from taggedpd import (AsNucleonStructure, ChristyBosted, ChristyBostedStructure, CombinedStructure,
                      FromNucleonStructure, GRSV2000Structure, Kinematics, Piecewise, Polarization,
                      SLACStructure, TaggedDIS, default_photoabsorption, inclusive_sf,
                      resonance_dis_model)
from taggedpd import christy_bosted as cb
from taggedpd.christy_bosted21 import PreliminaryFitWarning
from taggedpd.inclusive import Blend, InclusiveModel

X = np.linspace(0.02, 0.95, 40)
Q2 = np.linspace(1.0, 30.0, 40)


# ---------------------------------------------------------------------------
# adapters reproduce the providers they wrap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target_mass", [True, False])
@pytest.mark.parametrize("nuc", ["p", "n"])
def test_round_trip_slac(nuc, target_mass):
    slac = SLACStructure(R=0.18, target_mass=target_mass)
    rt = AsNucleonStructure(FromNucleonStructure(slac, Q2_min=0.0), target_mass=target_mass)
    for a, b in zip(slac.F1F2(X, Q2, nuc), rt.F1F2(X, Q2, nuc)):
        assert np.allclose(a, b, rtol=1e-12, atol=0)
    for a, b in zip(slac.FT_FL(X, Q2, nuc), rt.FT_FL(X, Q2, nuc)):
        assert np.allclose(a, b, rtol=1e-11, atol=1e-14)


def test_wrapped_slac_gives_identical_tagged_cross_section():
    """TaggedDIS with SLAC routed through the (W, Q^2) layer == the default (SLAC + GRSV)."""
    wrapped = CombinedStructure(AsNucleonStructure(FromNucleonStructure(SLACStructure(), Q2_min=0.0)),
                                GRSV2000Structure())
    m0, m1 = TaggedDIS("AV18"), TaggedDIS("AV18", structure=wrapped)
    x = np.array([0.05, 0.2, 0.4, 0.6]); q2 = np.array([2.0, 5.0, 10.0, 20.0])
    kin = Kinematics.from_y(x, q2, 0.6)
    pol = Polarization.beam(kin, "parallel", Pz=0.8, Pzz=0.6)
    a = m0.reduced_cross_section(kin, 1.05, 0.15, 0.3, lambda_e=0.5, pol=pol)
    b = m1.reduced_cross_section(kin, 1.05, 0.15, 0.3, lambda_e=0.5, pol=pol)
    assert np.allclose(a, b, rtol=1e-11)


def test_christy_bosted_structure_is_adapter():
    s = ChristyBostedStructure()
    assert isinstance(s, AsNucleonStructure)
    ref = AsNucleonStructure(ChristyBosted())
    x = np.array([0.2, 0.4, 0.6]); q2 = np.array([1.0, 2.0, 3.0])
    for nuc in "pn":
        assert np.allclose(np.array(s.FT_FL(x, q2, nuc)), np.array(ref.FT_FL(x, q2, nuc)), rtol=1e-14)
    # import paths kept: taggedpd.nucleon re-exports it lazily
    from taggedpd.nucleon import ChristyBostedStructure as C2
    assert C2 is ChristyBostedStructure


def test_aliases():
    assert taggedpd.photoproduction is inclusive_sf
    assert taggedpd.PhotoabsorptionModel is InclusiveModel


def test_as_nucleon_structure_kinematic_edges_and_outside_policy():
    s = AsNucleonStructure(default_photoabsorption())
    assert s.F1F2(np.array([0.0, 1.0, 1.2]), 2.0, "p")[0].tolist() == [0.0, 0.0, 0.0]
    # x = 0.05 at Q2 = 2: W ~ 6 GeV, outside every Christy-Bosted piece -> NaN by default ...
    assert np.isnan(s.F1F2(0.05, 2.0, "p")[0])
    # ... or 0 with outside="zero"
    s0 = AsNucleonStructure(default_photoabsorption(outside="zero"))
    assert s0.F1F2(0.05, 2.0, "p")[0] == 0.0


# ---------------------------------------------------------------------------
# resonance + DIS combination
# ---------------------------------------------------------------------------
def test_resonance_dis_pieces():
    m = resonance_dis_model()
    W = np.array([1.0, 1.08, 1.5, 2.4, 2.75, 3.05, 4.0])
    src = {q2: list(inclusive_sf(W, "p", "F2", Q2=q2, model=m, return_source=True)[1])
           for q2 in (0.0, 2.0, 9.0)}
    cbn, ext, thr, slac = ("Christy-Bosted", "Christy-Bosted (threshold extrapolation)",
                           "below pion threshold", "SLAC(R=0.18)")
    blend = "blend(Christy-Bosted->SLAC(R=0.18))"
    assert src[0.0] == [thr, ext, cbn, cbn, cbn, cbn, ""]          # no DIS / Regge at Q2 = 0
    assert src[2.0] == [thr, ext, cbn, cbn, blend, slac, slac]
    assert src[9.0] == [thr, slac, slac, slac, slac, slac, slac]    # above the CB Q2 range
    _, src_d = inclusive_sf(np.array([1.5, 2.75, 4.0]), "d", Q2=2.0, model=m, return_source=True)
    assert list(src_d) == [cbn, cbn, ""]                            # deuteron: CB only


@pytest.mark.parametrize("nuc", ["p", "n"])
def test_resonance_dis_continuity(nuc):
    m = resonance_dis_model()
    eps = 1e-9
    for Wb in (2.5, 3.0):
        for q2 in (1.0, 3.0, 6.0):
            for q in ("F_T", "F_L", "F2"):
                lo = inclusive_sf(Wb - eps, nuc, q, Q2=q2, model=m)
                hi = inclusive_sf(Wb + eps, nuc, q, Q2=q2, model=m)
                assert lo == pytest.approx(hi, rel=1e-6), (Wb, q2, q)


def test_resonance_dis_reproduces_pieces_in_tagged_dis():
    """In TaggedDIS: SLAC where W_n is above the window, Christy-Bosted where it is in the resonance region."""
    res = AsNucleonStructure(resonance_dis_model())
    slac, cbs = SLACStructure(), ChristyBostedStructure()
    q2 = 4.0
    x_dis = np.array([0.05, 0.1, 0.2])          # W ~ 8.8, 6.1, 4.1 GeV
    x_res = np.array([0.6, 0.7])                # W ~ 1.9, 1.6 GeV
    for nuc in "pn":
        assert np.allclose(res.FT_FL(x_dis, q2, nuc), slac.FT_FL(x_dis, q2, nuc), rtol=1e-11)
        assert np.allclose(res.FT_FL(x_res, q2, nuc), cbs.FT_FL(x_res, q2, nuc), rtol=1e-12)
    td = TaggedDIS("AV18", structure=CombinedStructure(res, GRSV2000Structure()))
    kin = Kinematics.from_y(np.array([0.05, 0.6]), q2, 0.6)
    assert np.all(np.isfinite(td.reduced_cross_section(kin, 1.0, 0.1, 0.3)))


def test_blend_domain_is_intersection():
    b = Blend(ChristyBosted(), FromNucleonStructure(SLACStructure(), Q2_min=1.0), 2.5, 3.0)
    assert b.domain(2.7, 2.0, "p") and not b.domain(2.7, 0.5, "p") and not b.domain(2.7, 9.0, "p")


def test_resonance_dis_2021_warns():
    with pytest.warns(PreliminaryFitWarning):
        m = resonance_dis_model(fit="2021")
    assert "PRELIMINARY" in m.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PreliminaryFitWarning)
        assert np.isfinite(inclusive_sf(2.0, "n", "F_L", Q2=2.0, model=m))


def test_lhapdf_like_provider_plugs_in():
    """Any NucleonStructure works as the DIS part (here a trivial analytic one)."""
    class Toy(taggedpd.NucleonStructure):
        name = "toy"
        target_mass = False

        def F1F2(self, x, Q2, nucleon="n"):
            x = np.asarray(x, float)
            F2 = 0.3 * np.ones(np.broadcast(x, np.asarray(Q2)).shape)
            return F2 / (2 * x * 1.2), F2

    m = resonance_dis_model(dis=Toy())
    assert inclusive_sf(4.0, "p", "F2", Q2=2.0, model=m) == pytest.approx(0.3)
    s = AsNucleonStructure(Piecewise([FromNucleonStructure(Toy())]), target_mass=False)
    assert np.allclose(s.FT_FL(0.1, 5.0, "p"), Toy().FT_FL(0.1, 5.0, "p"))
