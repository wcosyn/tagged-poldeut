r"""PRELIMINARY Christy 2021 resonance-region fits for p, n and d (port of SIMC ``F1F2IN21_v1.0.f``).

.. warning::
   **This parametrization is preliminary and unpublished.** Its own header calls it "F1F221
   version 1.0 prerelease 1 (April 4, 2022)" and describes the deuteron/neutron fit as
   "preliminary ... (to be published)". Do **not** use it for published results without
   checking its status with the author (M. E. Christy). For published work, use the 2009
   fits in :mod:`taggedpd.christy_bosted` (PRC 81, 055213 and PRC 77, 065206). Importing
   this module is harmless; constructing the photoabsorption model
   :class:`taggedpd.inclusive.ChristyBosted2021` emits a :class:`PreliminaryFitWarning`.

Origin
------
* Code: ``F1F2IN21_v1.0.f`` ("F1F221 v1.0 prerelease 1", M. E. Christy, JLab), as
  distributed with the Hall C simulation package SIMC,
  https://github.com/JeffersonLab/simc_gfortran (file added there in July 2022). The
  smearing tables ``f1f2tables/f{11,12,22}-onshell-cdbonn.dat`` shipped in
  ``taggedpd/data/f1f2tables/`` come from the same repository. The SIMC repository carries
  no license file; redistribution permission has not been confirmed.
* Fit authors according to the code header: proton fit, M. E. Christy (version
  "120521": JLab Hall data, SLAC DIS (Whitlow), and photoproduction data from DAPHNE and
  older sets); deuteron/neutron fit, M. E. Christy, N. Kalantarians, J. Ethier and
  W. Melnitchouk, "to be published", based on inclusive deuteron data and BONuS tagged
  n/d data, including photoproduction at Q^2 = 0. Stated validity: the header says
  W^2 < 32, Q^2 < 32 GeV^2; the routine comments say W^2 < 20 and Q^2 < 30 GeV^2.
* The functional form continues that of the 2009 fits (Christy & Bosted, PRC 81, 055213
  (2010); Bosted & Christy, PRC 77, 065206 (2008)), which should be cited as well.

Differences from the 2009 fits (:mod:`taggedpd.christy_bosted`)
-----------------------------------------------------------------
* **The neutron has its own fit** (``RESMODN``) with separate sigma_T and sigma_L; it is no
  longer 2(p+n)/2 - p, and no longer borrows the proton's R.
* **The deuteron is p + n smeared** with light-cone momentum distributions in the
  weak-binding approximation (CD-Bonn), tabulated in (gamma, y) (``SMEARSUB``; for
  Q^2 < 0.05 GeV^2 a variant ``SMEARSUBLOWQ`` that works at Q^2 = 0). There is no MEC term
  in the inelastic part, and off-shell corrections are switched off in this version.
* The proton is refitted (masses, widths, Q^2 dependence, new non-resonant form with
  pion and eta thresholds).

What is ported
--------------
``RESMODP``/``RESMODN`` (:func:`resmod21`), ``RESCSP``/``RESCSN`` (:func:`rescsp`,
:func:`rescsn`), ``SF`` (:func:`sf21`), ``GETFY`` (:func:`getfy`), ``DSIMPS``,
``SMEARSUB``/``SMEARSUBLOWQ``/``RESCSD`` (:func:`rescsd`), and the A <= 2 branches of
``F1F2IN21`` (:func:`f1f2in21`). Not ported: quasi-elastic (``F1F2QE21``, ``SQESUB``),
off-shell smearing (unused in this version), A > 2 (``SFCROSS``).

Faithfulness and quirks of the original (reproduced on purpose)
---------------------------------------------------------------
The port reproduces the FORTRAN compiled with double-precision literals to rounding, and
the FORTRAN as shipped (single-precision literals) to ~1e-6 for p, n (see tests). Things
the FORTRAN does that a user should know about:

1. ``SF`` computes F2 = (2xF1 + FL)/(1 + 4M^2x^2/Q^2), which is 0/0 at Q^2 = 0 (NaN in
   the FORTRAN). Here F2 = 0 is returned at Q^2 = 0 (its limit); F1 is unaffected.
2. ``GETFY`` interpolates the smearing table with bin index ``int(500 y) + 1``, i.e. it
   uses the *next* y bin pair, so the "interpolation" is a linear extrapolation from
   [y_k + 0.002, y_k + 0.004] back to y (a shift of up to one 0.002 bin).
3. Between ``SMEARSUBLOWQ`` (Q^2 < 0.05) and ``SMEARSUB`` the deuteron integrand
   changes from F1(z) to 2zF1(z)/(2x) = F1(z)/y, so the deuteron F1 is discontinuous at
   Q^2 = 0.05 GeV^2 (see the notebook for the size).
4. Smearing tables cover 1 <= gamma <= 5; beyond that the FORTRAN reads zero-padded
   common-block entries. Reproduced here (zero padding up to the FORTRAN's array size).

Units and conventions as in :mod:`taggedpd.christy_bosted`: GeV, microbarn, deuteron
F1, F2 **per nucleus** with F1 = M W1 (M the nucleon mass, x = Q^2/2M nu in (0, 2)),
W^2 computed as for a free proton (the routines use slightly different nucleon masses
internally, as in the FORTRAN). :mod:`taggedpd.inclusive` converts the deuteron F1 to
arXiv:2603.23700's convention M_D W1 with x_D = Q^2/2M_D nu.
"""

from __future__ import annotations

import gzip
import os

import numpy as np

__all__ = ["resmod21", "rescsp", "rescsn", "sf21", "getfy", "rescsd", "f1f2in21",
           "XVAL_P21", "XVAL_N21", "PreliminaryFitWarning", "VERSION"]

#: version string of the ported FORTRAN
VERSION = "F1F221 v1.0 prerelease 1 (April 4, 2022), SIMC F1F2IN21_v1.0.f"


class PreliminaryFitWarning(UserWarning):
    """Emitted when a preliminary (unpublished) parametrization is used."""


_DATA = os.path.join(os.path.dirname(__file__), "data", "f1f2tables")

_MPI = 0.134977
_META = 0.547862
_MP_P = 0.9382727         # RESMODP / RESCSP
_MP_N = 0.939565          # RESMODN
_MP_SF = 0.938272         # SF
_PI_SF = 3.14159          # SF, SMEARSUB (sic)
_ALPHA_SF = 1 / 137.03599
_MAVG = (0.938272 + 0.939565) / 2.0   # SMEARSUB, RESCSD: average nucleon mass
_EPSD = -2.2e-03          # deuteron binding in SMEARSUB [GeV]

#: RESCSP parameters (version 120521): 1-50 transverse, 51-100 longitudinal
XVAL_P21 = np.array([
    0.12291E+01, 0.15173E+01, 0.15044E+01, 0.17100E+01, 0.16801E+01,
    0.14312E+01, 0.12616E+00, 0.23000E+00, 0.92594E-01, 0.90606E-01,
    0.75000E-01, 0.35067E+00, 0.75729E+01, 0.56091E+01, 0.94606E+01,
    0.20156E+01, 0.66190E+01, 0.41732E+00, 0.23980E-01, 0.53136E+01,
    0.63752E+00, 0.11484E+02, 0.69949E-01, 0.26191E+01, 0.53603E-01,
    0.65000E+02, 0.15351E+00, 0.20624E+01, 0.23408E+01, 0.16100E+02,
    0.62414E+02, 0.17201E+01, 0.23261E+00, 0.65000E+02, 0.23292E+01,
    0.14980E+01, 0.23000E+00, 0.63385E+00, 0.19093E-01, 0.61061E-01,
    0.29146E-02, 0.54388E+00, 0.77997E+00, 0.28783E+00, 0.10605E+01,
    0.69793E+00, 0.20009E+01, 0.57000E+00, 0.41632E+01, 0.38427E+00,
    0.10000E+01, 0.99842E+00, 0.98719E+00, 0.10168E+01, 0.98945E+00,
    0.99594E+00, 0.98799E+00, 0.10271E+01, 0.10650E+01, 0.97920E+00,
    0.10152E+01, 0.99622E+00, 0.81011E+01, 0.10070E-02, 0.14857E+01,
    0.33445E+01, 0.31641E-09, 0.69755E+02, 0.55228E+01, 0.14438E+00,
    0.60474E+01, 0.65395E-07, 0.14129E+01, 0.58609E+00, 0.36220E+01,
    0.92699E+00, 0.14418E+01, 0.86403E-02, 0.10001E-03, 0.75106E+00,
    0.76077E+00, 0.42272E+00, 0.55511E-11, 0.52486E+00, 0.58153E+00,
    0.15798E+01, 0.50105E+00, 0.89149E+02, 0.72789E+00, 0.24813E-01,
    -.61906E+00, 0.10000E+01, 0.00000E+00, 0.00000E+00, 0.68158E+03,
    0.12429E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.10000E-05])

#: RESCSN parameters (November 16, 2021 version)
XVAL_N21 = np.array([
    0.12291E+01, 0.15173E+01, 0.15044E+01, 0.17100E+01, 0.16801E+01,
    0.14312E+01, 0.12616E+00, 0.23000E+00, 0.92594E-01, 0.90606E-01,
    0.75000E-01, 0.35067E+00, 0.69500E+01, 0.86633E+01, 0.11557E+02,
    0.22138E+01, 0.44886E+01, 0.20500E+03, 0.84433E+03, 0.31167E+01,
    0.96301E+00, 0.14956E+00, 0.20761E-07, 0.10440E+01, 0.40143E-03,
    0.90028E+02, 0.75248E-01, 0.20532E+00, 0.12444E-01, 0.34469E+03,
    0.19948E+00, 0.26925E+01, 0.48635E+01, 0.86000E+02, 0.67813E+04,
    0.44281E+02, 0.29548E+00, 0.65421E+00, 0.23787E-09, 0.51967E-01,
    0.39926E-08, 0.29960E+00, 0.97516E+00, 0.46934E-01, 0.14246E+03,
    0.55801E+00, 0.19349E+01, 0.27400E+00, 0.38891E+00, 0.40000E-02,
    0.10108E+01, 0.97020E+00, 0.98248E+00, 0.97768E+00, 0.10425E+01,
    0.10198E+01, 0.97822E+00, 0.98239E+00, 0.10103E+01, 0.10076E+01,
    0.10044E+01, 0.99687E+00, 0.16696E+01, 0.10721E-06, 0.54114E+00,
    0.11923E+04, 0.55938E+02, 0.95000E+03, 0.39840E+02, 0.22026E+03,
    0.30498E+01, 0.24459E+00, 0.95574E+00, 0.35596E+00, 0.21228E-05,
    0.96696E+01, 0.27563E+01, 0.93027E-01, 0.33559E+02, 0.31207E-01,
    0.29020E+02, 0.86417E+00, 0.36471E-08, 0.99167E+00, 0.68124E+00,
    0.10000E-01, 0.90227E-01, 0.40115E+01, 0.29915E+01, 0.45929E-01,
    -.16758E+01, 0.78493E+01, 0.78184E+01, 0.42074E+01, 0.41179E-05,
    0.80597E+00, 0.00000E+00, 0.00000E+00, 0.10045E+01, 0.62364E+00])

_ANG = np.array([1., 0., 2., 3., 0., 1., 3.])
_BR1 = np.array([1.00, 0.45, 0.60, 0.65, 0.60, 0.65, 0.60])   # single pion
_BR3 = np.array([0.0, 0.40, 0.08, 0.0, 0.20, 0.0, 0.0])       # eta
_BR2 = 1.0 - _BR1 - _BR3                                       # two pion


def _f(xval, i):
    return xval[i - 1]


def _split(xval):
    """(transverse, longitudinal) 50-parameter sets as built in RESCSP/RESCSN."""
    xt = xval[:50].copy()
    xl = xval[50:].copy()
    xl[:12] = xt[:12]
    xl[46:48] = xt[46:48]
    return xt, xl


_XP_T, _XP_L = _split(XVAL_P21)
_XN_T, _XN_L = _split(XVAL_N21)


# ===========================================================================
# RESMODP / RESMODN
# ===========================================================================
def resmod21(sf, w2, q2, xval, neutron=False):
    """sigma_T (sf=1) or sigma_L (sf=2) in microbarn; port of RESMODP (proton) / RESMODN (neutron)."""
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    mp = _MP_N if neutron else _MP_P
    mp2 = mp * mp
    W = np.sqrt(w2)
    q20 = _f(xval, 50)

    x0 = np.full(7, 0.160)
    if sf == 2:
        x0[0] = 0.07
    mon = 1.0 / (1.0 + q2 / 1.5) ** 1.0
    xb = q2 / (q2 + w2 - mp2)
    xpr1 = np.where(W <= mp + _MPI, 1.0, 1.0 / (1.0 + (w2 - (mp + _MPI) ** 2) / (q2 + q20)))
    xpr2 = np.where(W <= mp + _META, 1.0, 1.0 / (1.0 + (w2 - (mp + _META) ** 2) / (q2 + q20)))

    k = (w2 - mp2) / 2.0 / mp
    kcm = (w2 - mp2) / 2.0 / W
    epicm = (w2 + _MPI**2 - mp2) / 2.0 / W
    ppicm = np.sqrt(np.maximum(0.0, epicm**2 - _MPI**2))
    epi2cm = (w2 + (2.0 * _MPI) ** 2 - mp2) / 2.0 / W
    ppi2cm = np.sqrt(np.maximum(0.0, epi2cm**2 - (2.0 * _MPI) ** 2))
    eetacm = (w2 + _META * _META - mp2) / 2.0 / W
    petacm = np.sqrt(np.maximum(0.0, eetacm**2 - _META**2))

    mass = np.concatenate([xval[0:6], [_f(xval, 47)]])
    intw = np.concatenate([xval[6:12], [_f(xval, 48)]])

    heights = []
    for i in range(6):
        r1, r2, r3, r4 = xval[12 + 4 * i: 16 + 4 * i]
        if sf == 1:
            h = r1 * (1.0 + r2 * q2 / (1.0 + r3 * q2)) * mon**r4
        else:
            h = (r1 + r2 * q2) * np.exp(-1.0 * r3 * q2)
        heights.append(h * h)
    if sf == 2:
        if neutron:
            h7 = (_f(xval, 44) + _f(xval, 45) * q2) * np.exp(-1.0 * _f(xval, 46) * q2)
        else:
            h7 = (_f(xval, 16) + _f(xval, 20) * q2) * np.exp(-1.0 * _f(xval, 24) * q2)
    else:
        h7 = _f(xval, 49) * mon if neutron else _f(xval, 49) * mon ** _f(xval, 45)
    heights.append(h7 * h7)

    sig_res = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(7):
            m, iw, xx0, ang = mass[i], intw[i], x0[i], _ANG[i]
            kr = (m**2 - mp2) / 2.0 / mp
            kcmr = (m**2 - mp2) / 2.0 / m
            epicmr = (m**2 + _MPI**2 - mp2) / 2.0 / m
            ppicmr = np.sqrt(max(0.0, epicmr**2 - _MPI**2))
            epi2cmr = (m**2 + (2.0 * _MPI) ** 2 - mp2) / 2.0 / m
            ppi2cmr = np.sqrt(max(0.0, epi2cmr**2 - (2.0 * _MPI) ** 2))
            eetacmr = (m**2 + _META * _META - mp2) / 2.0 / m
            petacmr = np.sqrt(max(0.0, eetacmr**2 - _META**2))
            pw1 = iw * (ppicm / ppicmr) ** (2.0 * ang + 1.0) \
                * ((ppicmr**2 + xx0**2) / (ppicm**2 + xx0**2)) ** ang
            pw2 = iw * (ppi2cm / ppi2cmr) ** (2.0 * ang + 4.0) \
                * ((ppi2cmr**2 + xx0**2) / (ppi2cm**2 + xx0**2)) ** (ang + 2) * W / m
            pw3 = 0.0
            if i in (1, 4):
                pw3 = iw * (petacm / petacmr) ** (2.0 * ang + 1.0) \
                    * ((petacmr**2 + xx0**2) / (petacm**2 + xx0**2)) ** ang
            pgam = iw * (kcm / kcmr) ** 2 * (kcmr**2 + xx0**2) / (kcm**2 + xx0**2)
            width = _BR1[i] * pw1 + _BR2[i] * pw2 + _BR3[i] * pw3
            sigr = width * pgam / ((w2 - m**2) ** 2 + (m * width) ** 2)
            sig_res = sig_res + heights[i] * kr / k * kcmr / kcm * sigr / iw
        sig_res = sig_res * W
        if sf == 2:
            sig_res = sig_res * q2

        above = xpr1 < 1.0
        if sf == 1:
            A0 = _f(xval, 37) / (1.0 + q2 / _f(xval, 42)) ** _f(xval, 43)
            if neutron:
                t1 = _f(xval, 38) * np.log(1.05 + q2) + _f(xval, 39) / (1.05 + q2)
            else:
                t1 = _f(xval, 38) * np.log(1.06 + q2) + _f(xval, 39) / np.log(1.06 + q2)
            t2 = _f(xval, 40) * (1.0 + q2 / _f(xval, 41)) ** _f(xval, 44)
            nr = 389.4 * A0 * (1.0 - xpr1) ** t1 * xpr1**t2
            nr = nr + _f(xval, 46) * 389.4 * A0 * (1.0 - xpr2) ** t1 * xpr2**t2
        else:
            A0 = _f(xval, 37) / (1.0 + q2 / _f(xval, 39)) ** 2.0
            t1 = _f(xval, 38) / (1.0 + q2 / _f(xval, 40)) + _f(xval, 32) * np.log(q2 + _f(xval, 36))
            if neutron:
                t2 = _f(xval, 41) / (1.00 + q2 / _f(xval, 42)) ** _f(xval, 43)
            else:
                t2 = _f(xval, 41)
            nr = 389.4 * A0 * xb * (1.0 - xpr1) ** t1 * xpr1**t2
        sig_nr = np.where(above, nr, 0.0)
    return sig_res + sig_nr


def rescsp(w2, q2):
    """Proton (sigma_T, sigma_L) [microbarn] (port of RESCSP)."""
    return resmod21(1, w2, q2, _XP_T), resmod21(2, w2, q2, _XP_L)


def rescsn(w2, q2):
    """Neutron (sigma_T, sigma_L) [microbarn] (port of RESCSN)."""
    return resmod21(1, w2, q2, _XN_T, neutron=True), resmod21(2, w2, q2, _XN_L, neutron=True)


def sf21(w2, q2):
    """(F1p, FLp, F2p, F1n, FLn, F2n) from the cross sections (port of SF).

    FL is (1+gamma^2) F2 - 2x F1. At Q^2 = 0 the FORTRAN gives F2 = NaN (0/0); here F2 = 0.
    """
    w2 = np.asarray(w2, float)
    q2 = np.asarray(q2, float)
    mp2 = _MP_SF**2
    pi2 = _PI_SF * _PI_SF
    x = q2 / (q2 + w2 - mp2)
    sTp, sLp = rescsp(w2, q2)
    sTn, sLn = rescsn(w2, q2)
    conv = np.abs(w2 - mp2) / 0.3894e3 / pi2 / _ALPHA_SF / 8.0
    f1p, f1n = sTp * conv, sTn * conv
    fLp, fLn = sLp * 2.0 * x * conv, sLn * 2.0 * x * conv
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1.0 + 4.0 * mp2 * x * x / q2
        f2p = np.where(q2 > 0, (2.0 * x * f1p + fLp) / den, 0.0)
        f2n = np.where(q2 > 0, (2.0 * x * f1n + fLn) / den, 0.0)
    return f1p, fLp, f2p, f1n, fLn, f2n


# ===========================================================================
# smearing functions (GETFY) and Simpson (DSIMPS)
# ===========================================================================
_TABLES = None


def _load_tables():
    """CD-Bonn on-shell smearing tables, zero padded like the FORTRAN common block (50 x 1001)."""
    global _TABLES
    if _TABLES is None:
        out = {}
        for key, fname in (("11", "f11-onshell-cdbonn.dat.gz"), ("22", "f22-onshell-cdbonn.dat.gz"),
                           ("12", "f12-onshell-cdbonn.dat.gz")):
            with gzip.open(os.path.join(_DATA, fname), "rt") as fh:
                raw = np.loadtxt(fh)
            n = raw.shape[0]
            j = (np.arange(n) // 999)
            k = np.arange(n) - j * 999
            g = np.zeros(50)
            yv = np.zeros((50, 1001))
            fv = np.zeros((50, 1001))
            g[j] = raw[:, 0]
            yv[j, k] = raw[:, 1]
            fv[j, k] = raw[:, 2]
            out[key] = (g, yv, fv)
        _TABLES = out
    return _TABLES


def getfy(gamma, y):
    """(fy11, fy12, fy2) at (gamma, y): port of GETFY including its bin choice (see module notes)."""
    t = _load_tables()
    gamma = np.asarray(gamma, float)
    y = np.asarray(y, float)
    y = np.where(y >= 1.9999, 1.999, y)
    b = np.clip((y / 2.0 * 1000).astype(int), 0, 999)            # FORTRAN bin-1
    jl = np.clip(((gamma - 1.0) / 0.2).astype(int), 0, 48)        # FORTRAN jlow-1
    jh = jl + 1
    res = []
    for key in ("11", "12", "22"):
        g, yv, fv = t[key]
        fylow = (fv[jl, b + 1] * (y - yv[jl, b]) + fv[jl, b] * (yv[jl, b + 1] - y)) / 0.002
        fyhi = (fv[jh, b + 1] * (y - yv[jh, b]) + fv[jh, b] * (yv[jh, b + 1] - y)) / 0.002
        fy = (fylow * (g[jh] - gamma) + fyhi * (gamma - g[jl])) / 0.2
        res.append(np.maximum(fy, 0.0))
    fy11, fy12, fy2 = res
    return fy11, fy12, fy2


def _dsimps(F, a, b, n):
    """CERNLIB DSIMPS on F[0..n] (list of arrays), same summation order."""
    s1 = F[n - 1].copy()
    s2 = np.zeros_like(s1)
    for i in range(1, n - 2, 2):
        s1 = s1 + F[i]
        s2 = s2 + F[i + 1]
    s1 = s1 + s1 + s2
    return (F[0] + F[n] + s1 + s1) * (b - a) / (3 * n)


# ===========================================================================
# deuteron: SMEARSUB / SMEARSUBLOWQ / RESCSD
# ===========================================================================
def _smear(w2, q2, lowq):
    """Port of SMEARSUB (lowq=False) / SMEARSUBLOWQ (lowq=True); returns (f2s, f1s, fLs) per nucleus.

    All points in the call must share the same nbins (the caller groups them).
    """
    mp2 = _MAVG * _MAVG
    pi2 = _PI_SF * _PI_SF
    alpha = _ALPHA_SF
    n = len(w2)
    f1s = np.zeros(n); f2s = np.zeros(n); fLs = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = q2 / (w2 - mp2 + q2)
        if lowq:
            gamma2 = 1.0 + q2 * 4.0 * mp2 / (w2 + q2 - mp2) ** 2.0
        else:
            gamma2 = 1.0 + 4.0 * mp2 * x * x / q2
        gamma = np.sqrt(gamma2)
        ymin = np.maximum(x * (1.0 - 2.0 * _EPSD * _MAVG / q2), x)
        if lowq:
            ymin = np.where(q2 == 0.0, 0.001, ymin)
        ymax = np.minimum(1.0 + gamma2 / 2.0 + _EPSD / _MAVG, 2.0)
    ok = (x >= 0.0) & (ymin < ymax)
    if not np.any(ok):
        return f2s, f1s, fLs
    idx = np.nonzero(ok)[0]
    w2, q2, x, gamma, gamma2, ymin, ymax = (a[idx] for a in (w2, q2, x, gamma, gamma2, ymin, ymax))
    nb = _nbins(w2, q2)
    nbins = int(nb[0])
    inc = (ymax - ymin) / float(nbins)
    y = ymin - inc
    v1, v2 = [], []
    q0 = q2 == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(nbins + 1):
            y = y + inc
            z = x / y
            wwz = mp2 + q2 * (1.0 / z - 1.0)
            if lowq:
                t = 1.0 + 4.0 * mp2 * z * z / q2
                wwz = np.where(q0, mp2 + y * (w2 - mp2), wwz)
            sTp, sLp = rescsp(wwz, q2)
            sTn, sLn = rescsn(wwz, q2)
            conv = np.abs(wwz - mp2) / 0.3894e3 / pi2 / alpha / 8.0
            fLconv = 2 * q2 / np.abs(wwz - mp2 + q2) * conv
            if lowq:
                f1pi, f1ni = sTp * conv, sTn * conv
                fLpi, fLni = sLp * fLconv, sLn * fLconv
                f2pi = (2.0 * z * f1pi + fLpi) / t
                f2ni = (2.0 * z * f1ni + fLni) / t
            else:
                f1pi, f1ni = 2.0 * z * sTp * conv, 2.0 * z * sTn * conv
                fLpi, fLni = sLp * fLconv, sLn * fLconv
                f2pi = (f1pi + fLpi) / (1.0 + 4.0 * mp2 * z * z / q2)
                f2ni = (f1ni + fLni) / (1.0 + 4.0 * mp2 * z * z / q2)
            f2i = f2pi + f2ni
            f1i = f1pi + f1ni
            if lowq:
                f2i = np.where(q0, 0.0, f2i)
            fy11, fy12, fy2 = getfy(gamma, y)
            a1 = np.maximum(fy11 * f1i + fy12 * f2i, 0.0)
            a2 = np.maximum(fy2 * f2i, 0.0)
            # FORTRAN: "if(v.LT.0) v = 0" leaves NaN untouched; np.maximum propagates NaN too
            v1.append(a1)
            v2.append(a2)
        v1[0] = np.zeros_like(v1[0])
        v2[0] = np.zeros_like(v2[0])
        F1 = _dsimps(v1, ymin, ymax, nbins)
        F2 = _dsimps(v2, ymin, ymax, nbins)
        if not lowq:
            F1 = F1 / 2.0 / x
        FL = gamma2 * F2 - 2.0 * x * F1
    f1s[idx], f2s[idx], fLs[idx] = F1, F2, FL
    return f2s, f1s, fLs


def _nbins(w2, q2):
    nb = np.full(np.shape(w2), 60)
    nb = np.where(w2 < 1.6, 76, nb)
    nb = np.where((w2 >= 2.6) & (q2 >= 2.0), 48, nb)
    return nb


def rescsd(w2, q2):
    """Deuteron (F1, F2, FL) **per nucleus**: smeared p + n, no QE, no off-shell (port of RESCSD)."""
    w2, q2 = np.broadcast_arrays(np.asarray(w2, float), np.asarray(q2, float))
    shape = w2.shape
    w2, q2 = w2.ravel(), q2.ravel()
    F1 = np.zeros(w2.shape); F2 = np.zeros(w2.shape); FL = np.zeros(w2.shape)
    nb = _nbins(w2, q2)
    for lowq in (True, False):
        for nbv in np.unique(nb):
            sel = np.nonzero((nb == nbv) & ((q2 < 0.05) == lowq))[0]
            if sel.size == 0:
                continue
            f2s, f1s, fLs = _smear(w2[sel], q2[sel], lowq)
            F1[sel], F2[sel], FL[sel] = f1s, f2s, fLs
    return F1.reshape(shape), F2.reshape(shape), FL.reshape(shape)


# ===========================================================================
# driver
# ===========================================================================
def f1f2in21(target, q2, w2):
    """(F1, F2) for target "p", "n" (free nucleon) or "d" (whole deuteron); port of F1F2IN21 (A <= 2).

    PRELIMINARY fit, see module docstring. F2 = 0 at Q^2 = 0 (NaN in the FORTRAN).
    """
    w2, q2 = np.broadcast_arrays(np.asarray(w2, float), np.asarray(q2, float))
    if target in ("p", "n"):
        f1p, _, f2p, f1n, _, f2n = sf21(w2, q2)
        return (f1p, f2p) if target == "p" else (f1n, f2n)
    if target == "d":
        F1, F2, _ = rescsd(w2, q2)
        return F1, F2
    raise ValueError(f"target must be 'p', 'n' or 'd', got {target!r}")
