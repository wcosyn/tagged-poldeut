r"""Christy-Bosted resonance-region fits for the proton, neutron and deuteron.

Port of the FORTRAN package ``F1F209.f`` by P. E. Bosted and M. E. Christy
(version of 2009, as shipped in nucqcd ``Tools/F1F209_test.f``). Please
cite the original work when using this module:

* **Proton** -- M. E. Christy and P. E. Bosted, *Empirical fit to precision
  inclusive electron-proton cross sections in the resonance region*,
  Phys. Rev. C 81, 055213 (2010), arXiv:0712.3731.
  Fit range: 0 <= Q^2 < 8 GeV^2, 1.1 < W < 3.1 GeV; sigma_T and sigma_L are
  fitted separately, and real-photon (Q^2 = 0) data are included.
* **Deuteron and neutron** -- P. E. Bosted and M. E. Christy, *Empirical fit to
  inelastic electron-deuteron and electron-neutron resonance region transverse
  cross sections*, Phys. Rev. C 77, 065206 (2008), arXiv:0711.0159.
  Fit range: 0 <= Q^2 < 10 GeV^2, 1.2 < W < 3 GeV (below W = 1.2 GeV the
  nucleon average is constrained by MAID2007 pseudo-data).
* **Code** -- ``F1F209.f``, P. E. Bosted (JLab), distributed by the authors;
  the routine names below are those of the FORTRAN.

What is ported
--------------
=================  ===========================  ====================================
FORTRAN            here                          content
=================  ===========================  ====================================
``RESMOD507``      :func:`resmod507`             proton sigma_T (sf=1), sigma_L (sf=2) [ub]
``CHRISTY507``     :func:`christy507`            proton F1, R, sigma_T, sigma_L
``RESMODD``        :func:`resmodd`               F1 of the free (p+n)/2 average, from the deuteron fit
``MEC``            :func:`mec_deuteron`          empirical transverse "dip-region" term of the deuteron fit
``RESD``           :func:`resd`                  deuteron F1 / 2: Fermi-smeared resmodd + MEC
``PIND``           :func:`pind`                  Fermi-smeared proton (gives R for the deuteron)
``F1F2IN09``       :func:`f1f2in09`              driver for A <= 2: F1, F2, R of p, n, d
=================  ===========================  ====================================

Not ported: the A > 2 superscaling/EMC branch of ``F1F2IN09``, the quasi-elastic
``F1F2QE09`` (in particular, deuteron photodisintegration gamma d -> p n is *not*
part of the inelastic deuteron result), and ``RESDER`` (needs the fit's error
matrix file ``F1F207D2emat.dat``, which is not distributed with the code).

How the neutron is obtained
---------------------------
There is no separate neutron fit. Bosted & Christy fit the *free* nucleon
average :math:`\bar F_1 = (F_{1p}+F_{1n})/2` (``RESMODD``, same functional form as
the proton fit, its own parameters) such that, after Fermi smearing with the
deuteron momentum distribution (``RESD``, a plane-wave impulse approximation)
and adding an empirical MEC/dip-region term, it describes the inclusive
deuteron data. The neutron is then

.. math::  F_{1n}(W^2, Q^2) = 2\,\bar F_1(W^2, Q^2) - F_{1p}^{\rm Christy}(W^2, Q^2),

with W^2 computed as for a free nucleon at rest. Consequences:

* the neutron inherits the uncertainties of both fits, plus model dependence of
  the smearing/MEC used to unfold the deuteron; it is a derived quantity;
* the deuteron fit is **transverse only**: for the neutron (and the deuteron) the
  code uses the proton's R = sigma_L/sigma_T (Fermi-smeared for the deuteron);
* at Q^2 = 0 only F1 (= sigma_T) is meaningful; F2 and sigma_L vanish.

Kinematics and units
--------------------
GeV throughout. ``w2`` is the invariant mass squared *calculated assuming
scattering from a free nucleon at rest* (also for the deuteron, where it is just
a way to label the photon energy: nu = (w2 - M^2 + Q^2)/2M with the proton mass).
Cross sections are in microbarn. F1 is that of the free nucleon for p and n and of the
**whole deuteron** for d (as returned by ``F1F2IN09``), but always F1 = M W1
with the *nucleon* mass M (so x = Q^2/2M nu in (0, 2) for d). The inclusive layer
(:mod:`taggedpd.inclusive`) converts the deuteron to the paper's (Part II)
convention F1^D = M_D W1 with x_D = Q^2/2M_D nu. The relation
sigma_T = 4 pi^2 alpha F1 / (M K), with Hand's flux K = (W^2 - M^2)/2M, is the
one built into the fit (``christy507``).

Numerical faithfulness
----------------------
The FORTRAN assigns single-precision literals (e.g. ``mp = 0.9382727``, all
``data`` tables) to double variables. This port uses the literal values in double
precision. It agrees with the FORTRAN compiled with double-precision literals
(``-fdefault-real-8 -fdefault-double-8``) to ~1e-12, and with the FORTRAN as
shipped (which nucqcd links) to ~1e-7 typically, 7e-5 at worst.
``RESMODD`` evaluates its Breit-Wigner shapes from a table in 1 MeV bins of W;
:func:`resmodd` does the same by default (``binned=True``) so that the regression
tests are exact; ``binned=False`` evaluates them at the actual W (differences
are below a few 1e-3, at the steep sides of the resonances).

All functions are numpy-vectorized in ``w2`` and ``q2``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resmod507", "christy507", "resmodd", "mec_deuteron", "resd", "pind",
           "f1f2in09", "sigmaT_from_F1", "F1_from_sigmaT",
           "MP_CB", "W2_MIN", "UB_GEV2", "XVAL_PROTON", "XVAL_DEUTERON"]

# ---------------------------------------------------------------------------
# constants as used in the FORTRAN (deliberately not the package constants)
# ---------------------------------------------------------------------------
#: proton mass in RESMOD507 / CHRISTY507 / RESMODD
MP_CB = 0.9382727
#: nucleon mass in PIND / RESD / MEC
_AM = 0.9383
#: proton mass in F1F2IN09 (used for W1 <-> F1 and nu)
_PM = 0.93828
#: deuteron mass in PIND / RESD
_AMD = 1.8756
_MPI = 0.135
_META = 0.547
_PI = 3.141593            # sic, as in the FORTRAN
_ALPHA = 1.0 / 137.036
#: 1 GeV^-2 in microbarn, as in the FORTRAN (0.3894e3)
UB_GEV2 = 0.3894e3
#: F1F2IN09 / PIND / RESD only call the nucleon fits above this W^2 [GeV^2]
#: (just above the pion-production threshold (m_p + m_pi)^2 = 1.1519 GeV^2)
W2_MIN = 1.155

# ---------------------------------------------------------------------------
# fit parameters
# ---------------------------------------------------------------------------
#: CHRISTY507 parameters: 1-50 transverse, 51-100 longitudinal (FORTRAN ``xval``)
XVAL_PROTON = np.array([
    0.12298E+01, 0.15304E+01, 0.15057E+01, 0.16980E+01, 0.16650E+01,
    0.14333E+01, 0.13573E+00, 0.22000E+00, 0.82956E-01, 0.95782E-01,
    0.10936E+00, 0.37944E+00, 0.77805E+01, 0.42291E+01, 0.12598E+01,
    0.21242E+01, 0.63351E+01, 0.68232E+04, 0.33521E+05, 0.25686E+01,
    0.60347E+00, 0.21240E+02, 0.55746E-01, 0.24886E+01, 0.23305E+01,
    -.28789E+00, 0.18607E+00, 0.63534E-01, 0.19790E+01, -.56175E+00,
    0.38964E+00, 0.54883E+00, 0.22506E-01, 0.46213E+03, 0.19221E+00,
    0.19141E+01, 0.24606E+03, 0.67469E-01, 0.13501E+01, 0.12054E+00,
    -.89360E+02, 0.20977E+00, 0.15715E+01, 0.90736E-01, -.38495E-02,
    0.10362E-01, 0.19341E+01, 0.38000E+00, 0.34187E+01, 0.14462E+00,
    0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
    0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
    0.00000E+00, 0.00000E+00, 0.29414E+02, 0.19910E+02, 0.22587E+00,
    0.00000E+00, 0.00000E+00, 0.38565E+04, 0.65717E+00, 0.00000E+00,
    0.15792E+03, 0.97046E+02, 0.31042E+00, 0.00000E+00, 0.42160E+01,
    0.38200E-01, 0.12182E+01, 0.00000E+00, 0.13764E+02, 0.31393E+00,
    0.29997E+01, 0.00000E+00, 0.55124E+01, 0.53743E-01, 0.13091E+01,
    0.00000E+00, 0.86746E+02, 0.40864E-04, 0.40294E+01, 0.31285E+01,
    0.33403E+00, 0.49623E+01, 0.00000E+00, 0.00000E+00, 0.11000E+02,
    0.18951E+01, 0.51376E+00, 0.00000E+00, 0.42802E+01, 0.00000E+00])

#: deuteron-fit parameters (FORTRAN ``xvald0`` in F1F2IN09)
XVAL_DEUTERON = np.array([
    0.1964E+01, 0.1086E+01, 0.5313E-02, 0.1265E+01, 0.8000E+01,
    0.2979E+00, 0.1354E+00, 0.2200E+00, 0.8296E-01, 0.9578E-01,
    0.1094E+00, 0.3794E+00, 0.8122E+01, 0.5189E+01, 0.3290E+01,
    0.1870E+01, 0.6110E+01, -0.3464E+02, 0.9000E+03, 0.1717E+01,
    0.4335E-01, 0.1915E+03, 0.2232E+00, 0.2119E+01, 0.2088E+01,
    -0.3029E+00, 0.2012E+00, 0.1104E-02, 0.2276E-01, -0.4562E+00,
    0.2397E+00, 0.1204E+01, 0.2321E-01, 0.5419E+03, 0.2247E+00,
    0.2168E+01, 0.2266E+03, 0.7649E-01, 0.1457E+01, 0.1318E+00,
    -0.7534E+02, 0.1776E+00, 0.1636E+01, 0.1350E+00, -0.5596E-02,
    0.5883E-02, 0.1934E+01, 0.3800E+00, 0.3319E+01, 0.1446E+00])

#: RESMODD: resonance masses and widths are fixed to these "5/07" values
_XVAL0_D = np.array([
    0.12298E+01, 0.15304E+01, 0.15057E+01, 0.16980E+01, 0.16650E+01,
    0.14333E+01, 0.13573E+00, 0.22000E+00, 0.82956E-01, 0.95782E-01,
    0.10936E+00, 0.37944E+00])

# resonances: P33(1232), S11(1535), D13(1520), F15(1680), S11(1650), P11(1440), 7th
_ANG = np.array([1., 0., 2., 3., 0., 1., 3.])      # meson angular momentum


def _f(xval, i):
    """FORTRAN 1-based parameter access."""
    return xval[i - 1]


def _cm_momenta(W, w2, mp):
    """photon (k, kcm) and meson c.m. momenta used in the threshold Breit-Wigners."""
    mp2 = mp * mp
    k = (w2 - mp2) / 2.0 / mp
    kcm = (w2 - mp2) / 2.0 / W
    epicm = (w2 + _MPI**2 - mp2) / 2.0 / W
    ppicm = np.sqrt(np.maximum(0.0, epicm**2 - _MPI**2))
    epi2cm = (w2 + (2.0 * _MPI) ** 2 - mp2) / 2.0 / W
    ppi2cm = np.sqrt(np.maximum(0.0, epi2cm**2 - (2.0 * _MPI) ** 2))
    eetacm = (w2 + _META * _META - mp2) / 2.0 / W
    petacm = np.sqrt(np.maximum(0.0, eetacm**2 - _META**2))
    return k, kcm, ppicm, ppi2cm, petacm


def _res_momenta(mass, mp):
    """the same momenta at W = resonance mass (scalars)."""
    mp2 = mp * mp
    kr = (mass**2 - mp2) / 2.0 / mp
    kcmr = (mass**2 - mp2) / 2.0 / mass
    epicmr = (mass**2 + _MPI**2 - mp2) / 2.0 / mass
    ppicmr = np.sqrt(max(0.0, epicmr**2 - _MPI**2))
    epi2cmr = (mass**2 + (2.0 * _MPI) ** 2 - mp2) / 2.0 / mass
    ppi2cmr = np.sqrt(max(0.0, epi2cmr**2 - (2.0 * _MPI) ** 2))
    eetacmr = (mass**2 + _META * _META - mp2) / 2.0 / mass
    petacmr = np.sqrt(max(0.0, eetacmr**2 - _META**2))
    return kr, kcmr, ppicmr, ppi2cmr, petacmr


def _partial(intw, p, pr, x0, ang):
    """1-pion / eta partial width: Gamma (p/p_r)^(2l+1) ((p_r^2+X0^2)/(p^2+X0^2))^l."""
    return intw * (p / pr) ** (2.0 * ang + 1.0) * ((pr**2 + x0**2) / (p**2 + x0**2)) ** ang


def _partial2pi(intw, p, pr, x0, ang, W, mass):
    """2-pion partial width (as in RESMOD507)."""
    return (intw * (p / pr) ** (2.0 * ang + 4.0)
            * ((pr**2 + x0**2) / (p**2 + x0**2)) ** (ang + 2.0) * W / mass)


# ===========================================================================
# proton: RESMOD507 / CHRISTY507
# ===========================================================================
def resmod507(sf, w2, q2, xval):
    """Proton sigma_T (``sf=1``) or sigma_L (``sf=2``) in microbarn (port of RESMOD507).

    ``xval`` is the 50-parameter set for that structure function, as built in
    :func:`christy507`. Valid for w2 above the pion threshold.
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    mp = MP_CB
    mp2 = mp * mp
    W = np.sqrt(w2)
    wdif1 = W - (mp + _MPI)

    if sf == 1:
        m0, q20 = 0.125, 0.05
    else:
        m0, q20 = _f(xval, 49), 0.125

    br1 = np.array([1.0, 0.45, 0.65, 0.65, 0.4, 0.65, 0.50])   # single pion
    br3 = np.array([0.0, 0.45, 0.0, 0.0, 0.1, 0.0, 0.0])       # eta
    br2 = 1.0 - br1 - br3                                       # two pion
    x0 = np.full(7, 0.215)
    x0[0] = _f(xval, 50)

    xb = q2 / (q2 + w2 - mp2)
    xpr1 = 1.0 / (1.0 + (w2 - (mp + _MPI) ** 2) / (q2 + q20))
    t = np.log(np.log((q2 + m0) / 0.330**2) / np.log(m0 / 0.330**2))

    k, kcm, ppicm, ppi2cm, petacm = _cm_momenta(W, w2, mp)

    mass = np.empty(7)
    intwidth = np.empty(7)
    mass[:6] = xval[0:6]
    intwidth[:6] = xval[6:12]
    if sf == 2:
        mass[6], intwidth[6] = _f(xval, 43), _f(xval, 44)
    else:
        mass[6], intwidth[6] = _f(xval, 47), _f(xval, 48)

    # resonance heights (Q2 dependence); rescoef(i,j) = xval(12 + 4(i-1) + j)
    heights = []
    for i in range(6):
        r1, r2, r3, r4 = xval[12 + 4 * i: 16 + 4 * i]
        if sf == 1:
            h = r1 * (1.0 + r2 * q2 / (1.0 + r3 * q2)) / (1.0 + q2 / 0.91) ** r4
        else:
            h = r1 * q2 / (1.0 + r2 * q2) * np.exp(-1.0 * r3 * q2)
        heights.append(h * h)
    if sf == 2:
        h7 = _f(xval, 45) * q2 / (1.0 + _f(xval, 46) * q2) * np.exp(-1.0 * _f(xval, 47) * q2)
    else:
        h7 = _f(xval, 49) / (1.0 + q2 / 0.91) ** 1.0
    heights.append(h7 * h7)

    sig_res = 0.0
    for i in range(7):
        kr, kcmr, ppicmr, ppi2cmr, petacmr = _res_momenta(mass[i], mp)
        pw1 = _partial(intwidth[i], ppicm, ppicmr, x0[i], _ANG[i])
        pw2 = _partial2pi(intwidth[i], ppi2cm, ppi2cmr, x0[i], _ANG[i], W, mass[i])
        pw3 = _partial(intwidth[i], petacm, petacmr, x0[i], _ANG[i]) if i in (1, 4) else 0.0
        pgam = intwidth[i] * (kcm / kcmr) ** 2 * (kcmr**2 + x0[i] ** 2) / (kcm**2 + x0[i] ** 2)
        width = br1[i] * pw1 + br2[i] * pw2 + br3[i] * pw3
        sigr = width * pgam / ((w2 - mass[i] ** 2) ** 2 + (mass[i] * width) ** 2)
        sig_res = sig_res + heights[i] * kr / k * kcmr / kcm * sigr / intwidth[i]
    sig_res = sig_res * W

    # non-resonant background; nr_coef(i,j) = xval(36 + 4(i-1) + j)
    if sf == 1:
        sig_nr = 0.0
        for i in (1, 2):
            c1, c2, c3, c4 = xval[36 + 4 * (i - 1): 40 + 4 * (i - 1)]
            h_nr = c1 / (q2 + c2) ** (c3 + c4 * q2 + _f(xval, 44 + i) * q2**2)
            sig_nr = sig_nr + h_nr * np.maximum(wdif1, 0.0) ** ((2 * i + 1) / 2.0)
        sig_nr = sig_nr * xpr1
    else:
        c1, c2, c3, c4 = xval[36:40]
        sig_nr = (c1 * (1.0 - xpr1) ** (c3 + c2 * t) / (1.0 - xb) / (q2 + q20)
                  * (q2 / (q2 + q20)) ** c4 * xpr1 ** (_f(xval, 41) + _f(xval, 42) * t))
    return sig_res + sig_nr


_XVAL_T = XVAL_PROTON[:50].copy()
_XVAL_L = XVAL_PROTON[50:].copy()
_XVAL_L[:12] = _XVAL_T[:12]
_XVAL_L[42] = _XVAL_T[46]     # xvalL(43) = xval1(47)
_XVAL_L[43] = _XVAL_T[47]     # xvalL(44) = xval1(48)
_XVAL_L[49] = _XVAL_T[49]     # xvalL(50) = xval1(50)


def christy507(w2, q2):
    """Proton fit (port of CHRISTY507). Returns (F1, R, sigma_T, sigma_L), sigma in microbarn.

    No range check (like the FORTRAN); callers should require w2 > :data:`W2_MIN`.
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    mp2 = MP_CB**2
    sigT = resmod507(1, w2, q2, _XVAL_T)
    sigL = resmod507(2, w2, q2, _XVAL_L)
    F1 = sigT * (w2 - mp2) / 8.0 / _PI / _PI / _ALPHA / UB_GEV2
    with np.errstate(divide="ignore", invalid="ignore"):
        R = sigL / sigT
    return F1, R, sigT, sigL


# ===========================================================================
# free nucleon average from the deuteron fit: RESMODD
# ===========================================================================
_D_MASS = np.concatenate([_XVAL0_D[:6], [1.9341]])
_D_INTW = np.concatenate([_XVAL0_D[6:12], [0.380]])
_D_BR1 = np.array([1.0, 0.5, 0.65, 0.65, 0.4, 0.65, 0.6])
_D_X0 = np.array([0.1446] + [0.215] * 6)


def _resmodd_shapes(W):
    """The 7 Breit-Wigner shapes ``sigr(iw, i)`` of RESMODD at c.m. energy W (array)."""
    mp = MP_CB
    w2 = W * W
    k, kcm, ppicm, ppi2cm, petacm = _cm_momenta(W, w2, mp)
    out = []
    for i in range(7):
        mass, intw, x0, ang = _D_MASS[i], _D_INTW[i], _D_X0[i], _ANG[i]
        kr, kcmr, ppicmr, ppi2cmr, petacmr = _res_momenta(mass, mp)
        pw1 = _partial(intw, ppicm, ppicmr, x0, ang)
        if i != 1:
            pw2 = _partial2pi(intw, ppi2cm, ppi2cmr, x0, ang, W, mass)
        else:     # S11(1535): second channel is eta
            pw2 = _partial(intw, petacm, petacmr, x0, ang)
        pgam = intw * (kcm / kcmr) ** 2 * (kcmr**2 + x0**2) / (kcm**2 + x0**2)
        width = _D_BR1[i] * pw1 + (1.0 - _D_BR1[i]) * pw2
        out.append(width * pgam / ((w2 - mass**2) ** 2 + (mass * width) ** 2)
                   * kr / k * kcmr / kcm / intw)
    return out


def resmodd(w2, q2, xval=XVAL_DEUTERON, binned=True):
    """F1 of the free-nucleon average (F1p + F1n)/2 from the deuteron fit (port of RESMODD).

    Returns 0 outside 1.07327^2 <= w2 <= 25, 0 <= q2 <= 11 (the FORTRAN prints an
    error message there). ``binned=True`` reproduces the FORTRAN's 1 MeV W table
    for the resonance shapes; ``False`` evaluates them at the exact W.
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    mp = MP_CB
    mp2 = mp * mp
    ok = (w2 >= 1.07327**2) & (w2 <= 25.0) & (q2 >= 0.0) & (q2 <= 11.0)
    w2s = np.where(ok, w2, 1.5)          # safe dummy outside the range
    q2s = np.where(ok, q2, 0.0)
    W = np.sqrt(w2s)

    if binned:
        iw = np.floor(1000.0 * W)
        shapes = _resmodd_shapes(0.001 * (iw + 0.5))
    else:
        shapes = _resmodd_shapes(W)

    dip = 1.0 / (1.0 + q2s / 0.91) ** 2
    sig_res = 0.0
    for i in range(6):
        r1, r2, r3, r4 = xval[12 + 4 * i: 16 + 4 * i]
        h = r1 * (1.0 + r2 * q2s / (1.0 + r3 * q2s)) / (1.0 + q2s / 0.91) ** r4
        sig_res = sig_res + h**2 * shapes[i]
    h7 = _f(xval, 49) * dip
    sig_res = (sig_res + h7**2 * shapes[6]) * W

    xpr = 1.0 / (1.0 + (w2s - (mp + _MPI) ** 2) / (q2s + 0.05))
    wdif = np.maximum(W - (mp + _MPI), 0.0)
    sig_nr = 0.0
    for i in (1, 2):
        c1, c2, c3, c4 = xval[36 + 4 * (i - 1): 40 + 4 * (i - 1)]
        sig_nr = sig_nr + (c1 * wdif ** ((2 * i + 1) / 2.0)
                           / (q2s + c2) ** (c3 + c4 * q2s + _f(xval, 44 + i) * q2s**2))
    sig_nr = sig_nr * xpr

    F1 = (sig_res + sig_nr) * (w2s - mp2) / 8.0 / _PI / _PI / _ALPHA / UB_GEV2
    return np.where(ok, F1, 0.0)


# ===========================================================================
# deuteron: MEC, RESD, PIND
# ===========================================================================
def mec_deuteron(q2, w2, xval=XVAL_DEUTERON):
    """Empirical transverse dip-region term of the deuteron fit, contribution to F1_d / 2 (port of MEC).

    Returns 0 where nu <= 0 (the FORTRAN would produce NaN there).
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    W = np.sqrt(np.maximum(w2, 0.0))
    nu = (w2 - _AM**2 + q2) / 2.0 / _AM
    nus = np.where(nu > 0, nu, 1.0)
    f1 = (_f(xval, 1) * np.exp(-(W - _f(xval, 2)) ** 2 / _f(xval, 3))
          / (1.0 + np.maximum(0.3, q2) / _f(xval, 4)) ** _f(xval, 5) * nus ** _f(xval, 6))
    return np.where(nu > 0, f1, 0.0)


# Fermi-smearing tables of PIND / RESD (deuteron momentum distribution).
# Coarse: 20 bins in p_z with <p_z>, <p^2> and weights (used for w2 > 1.30).
_FYD = np.array([0.4965, 0.4988, 0.4958, 0.5008, 0.5027, 0.5041, 0.5029, 0.5034, 0.4993, 0.5147,
                 0.5140, 0.4975, 0.5007, 0.4992, 0.4994, 0.4977, 0.5023, 0.4964, 0.4966, 0.4767])
_AVPZ = np.array([-0.1820, -0.0829, -0.0590, -0.0448, -0.0345, -0.0264, -0.0195, -0.0135, -0.0079,
                  -0.0025, 0.0029, 0.0083, 0.0139, 0.0199, 0.0268, 0.0349, 0.0453, 0.0598, 0.0844,
                  0.1853])
_AVP2 = np.array([0.0938, 0.0219, 0.0137, 0.0101, 0.0081, 0.0068, 0.0060, 0.0054, 0.0051, 0.0049,
                  0.0050, 0.0051, 0.0055, 0.0060, 0.0069, 0.0081, 0.0102, 0.0140, 0.0225, 0.0964])
# Fine: 200 bins p_z = -1 + 0.01 (i - 1/2), symmetric tables (used near/below threshold)
_FYDF_HALF = np.array([
    0.00001, 0.00002, 0.00003, 0.00005, 0.00006, 0.00009, 0.00010, 0.00013,
    0.00015, 0.00019, 0.00021, 0.00026, 0.00029, 0.00034, 0.00038, 0.00044,
    0.00049, 0.00057, 0.00062, 0.00071, 0.00078, 0.00089, 0.00097, 0.00109,
    0.00119, 0.00134, 0.00146, 0.00161, 0.00176, 0.00195, 0.00211, 0.00232,
    0.00252, 0.00276, 0.00299, 0.00326, 0.00352, 0.00383, 0.00412, 0.00447,
    0.00482, 0.00521, 0.00560, 0.00603, 0.00648, 0.00698, 0.00747, 0.00803,
    0.00859, 0.00921, 0.00985, 0.01056, 0.01126, 0.01205, 0.01286, 0.01376,
    0.01467, 0.01569, 0.01671, 0.01793, 0.01912, 0.02049, 0.02196, 0.02356,
    0.02525, 0.02723, 0.02939, 0.03179, 0.03453, 0.03764, 0.04116, 0.04533,
    0.05004, 0.05565, 0.06232, 0.07015, 0.07965, 0.09093, 0.10486, 0.12185,
    0.14268, 0.16860, 0.20074, 0.24129, 0.29201, 0.35713, 0.44012, 0.54757,
    0.68665, 0.86965, 1.11199, 1.43242, 1.86532, 2.44703, 3.22681, 4.24972,
    5.54382, 7.04016, 8.48123, 9.40627])
_FYDF = np.concatenate([_FYDF_HALF, _FYDF_HALF[::-1]])
_AVP2F_HALF = np.array([
    1.0, 0.98974, 0.96975, 0.96768, 0.94782, 0.94450, 0.92494, 0.92047,
    0.90090, 0.89563, 0.87644, 0.87018, 0.85145, 0.84434, 0.82593, 0.81841,
    0.80021, 0.79212, 0.77444, 0.76553, 0.74866, 0.73945, 0.72264, 0.71343,
    0.69703, 0.68740, 0.67149, 0.66182, 0.64631, 0.63630, 0.62125, 0.61154,
    0.59671, 0.58686, 0.57241, 0.56283, 0.54866, 0.53889, 0.52528, 0.51581,
    0.50236, 0.49291, 0.47997, 0.47063, 0.45803, 0.44867, 0.43665, 0.42744,
    0.41554, 0.40656, 0.39511, 0.38589, 0.37488, 0.36611, 0.35516, 0.34647,
    0.33571, 0.32704, 0.31656, 0.30783, 0.29741, 0.28870, 0.27820, 0.26945,
    0.25898, 0.25010, 0.23945, 0.23023, 0.21943, 0.20999, 0.19891, 0.18911,
    0.17795, 0.16793, 0.15669, 0.14667, 0.13553, 0.12569, 0.11504, 0.10550,
    0.09557, 0.08674, 0.07774, 0.06974, 0.06184, 0.05484, 0.04802, 0.04203,
    0.03629, 0.03129, 0.02654, 0.02247, 0.01867, 0.01545, 0.01251, 0.01015,
    0.00810, 0.00664, 0.00541, 0.00512])
_AVP2F = np.concatenate([_AVP2F_HALF, _AVP2F_HALF[::-1]])
_PZF = -1.0 + 0.01 * (np.arange(1, 201) - 0.5)


def _smear(w2, q2, fn):
    """Fermi smearing of PIND/RESD: sum_i weight_i * fn(w2'_i, q2) over w2'_i > W2_MIN.

    ``fn`` returns a tuple of arrays; the tuple of smeared arrays is returned.
    """
    w2, q2 = np.broadcast_arrays(np.asarray(w2, float), np.asarray(q2, float))
    shape = w2.shape
    w2, q2 = w2.ravel(), q2.ravel()
    nu = (w2 - _AM**2 + q2) / 2.0 / _AM
    qv = np.sqrt(nu**2 + q2)
    coarse = w2 > 1.30
    acc = None
    # bins as (p2, pz, weight); coarse (20 bins) above w2 = 1.30, fine (200 bins) below
    for branch, bins in ((coarse, zip(_AVP2, _AVPZ, _FYD / 10.0)),
                         (~coarse, zip(_AVP2F, _PZF, _FYDF / 100.0))):
        idx = np.nonzero(branch)[0]
        if idx.size == 0:
            continue
        nu_b, qv_b, q2_b = nu[idx], qv[idx], q2[idx]
        for p2, pz, wt in bins:
            w2p = (_AMD + nu_b - np.sqrt(_AM**2 + p2)) ** 2 - qv_b**2 + 2.0 * qv_b * pz - p2
            use = w2p > W2_MIN
            if not np.any(use):
                continue
            vals = fn(w2p[use], q2_b[use])
            if acc is None:
                acc = [np.zeros(w2.shape) for _ in vals]
            for a, v in zip(acc, vals):
                a[idx[use]] += v * wt
    if acc is None:
        return None
    return [a.reshape(shape) for a in acc]


def resd(q2, w2, xval=XVAL_DEUTERON, binned=True):
    """Deuteron F1 divided by 2 (Fortran convention F1 = M W1): Fermi-smeared :func:`resmodd` plus :func:`mec_deuteron` (port of RESD)."""
    acc = _smear(w2, q2, lambda a, b: (resmodd(a, b, xval, binned),))
    f1 = np.zeros(np.broadcast(np.asarray(w2), np.asarray(q2)).shape) if acc is None else acc[0]
    return f1 + mec_deuteron(q2, w2, xval)


def pind(w2, q2):
    """Proton Fermi-smeared with the deuteron momentum distribution (port of PIND).

    Returns (F1, R, sigma_T, sigma_L); used for R of the deuteron.
    """
    acc = _smear(w2, q2, lambda a, b: (lambda F1, R, sT, sL: (F1, sT, sL))(*christy507(a, b)))
    shape = np.broadcast(np.asarray(w2), np.asarray(q2)).shape
    if acc is None:
        z = np.zeros(shape)
        return z, z.copy(), z.copy(), z.copy()
    F1, sigt, sigl = acc
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(sigt != 0.0, sigl / sigt, 0.0)
    return F1, R, sigt, sigl


# ===========================================================================
# driver: F1F2IN09 for A <= 2
# ===========================================================================
def f1f2in09(target, q2, w2, binned=True):
    """F1, F2, R for ``target`` in {"p", "n", "d"} (port of F1F2IN09, A <= 2 branches).

    p : Christy-Bosted proton fit.
    n : 2 * resmodd - christy507 (see module docstring); R of the proton.
    d : 2 * resd, i.e. per **nucleus**; R from the Fermi-smeared proton (pind).
    F2 = nu W2 with W2 = W1 (1+R)/(1 + nu^2/Q^2); F2 = 0 at Q^2 = 0.
    For p and n everything is 0 for w2 <= :data:`W2_MIN` (below the pion threshold).
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    w2, q2 = np.broadcast_arrays(w2, q2)
    nu = (w2 - _PM**2 + q2) / 2.0 / _PM
    if target in ("p", "n"):
        above = w2 > W2_MIN
        w2s = np.where(above, w2, 2.0)
        F1, R, _, _ = christy507(w2s, q2)
        if target == "n":
            F1 = 2.0 * resmodd(w2s, q2, binned=binned) - F1
        F1 = np.where(above, F1, 0.0)
        R = np.where(above, R, 0.0)
    elif target == "d":
        F1 = 2.0 * resd(q2, w2, binned=binned)
        R = pind(w2, q2)[1]
    else:
        raise ValueError(f"target must be 'p', 'n' or 'd', got {target!r}")
    W1 = F1 / _PM
    with np.errstate(divide="ignore", invalid="ignore"):
        W2 = np.where(q2 > 0, W1 * (1.0 + R) / (1.0 + nu**2 / q2), 0.0)
    return F1, nu * W2, R


# ===========================================================================
# F1 <-> sigma_T
# ===========================================================================
def sigmaT_from_F1(F1, w2, mass=MP_CB):
    """sigma_T [microbarn] = 8 pi^2 alpha F1 / (W^2 - M^2) (Hand convention, as in the fit)."""
    w2 = np.asarray(w2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w2 > mass**2, F1 * 8.0 * _PI**2 * _ALPHA * UB_GEV2 / (w2 - mass**2), 0.0)


def F1_from_sigmaT(sigT, w2, mass=MP_CB):
    """Inverse of :func:`sigmaT_from_F1`."""
    w2 = np.asarray(w2, float)
    return sigT * (w2 - mass**2) / (8.0 * _PI**2 * _ALPHA * UB_GEV2)
