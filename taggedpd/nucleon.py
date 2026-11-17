r"""Nucleon structure functions (unpolarized F1, F2 and polarized g1, g2).

Design
------
The tagged-DIS code only needs, for the *active* nucleon, the four functions

* :math:`F_{[UU,T]n} = 2F_1`, :math:`F_{[UU,L]n} = F_2/x - 2F_1` (paper Eq. (5.10)),
* :math:`F_{[LS_L]n} = 2 g_1`, :math:`F_{[LS_T]n} = -2\gamma_n (g_1+g_2)` (Eq. (5.24)),

evaluated at the nucleon-level variables (x_n, Q^2). They are provided by an
object implementing the :class:`NucleonStructure` interface:

``F1F2(x, Q2, nucleon)``, ``g1(x, Q2, nucleon)``, ``g1_plus_g2(x, Q2, nucleon)``

with ``nucleon`` = ``"n"`` or ``"p"``; all inputs are numpy-broadcastable, Q2 in
GeV^2. Three implementations are shipped:

:class:`SLACStructure`
    Unpolarized F2 from the SLAC/Bodek fit [Bodek et al., PRD 20 (1979) 1471]
    (port of ``NucleonStructure::f2n_b/f2p_b/bodek``), F1 from F2 and
    R = sigma_L/sigma_T (constant R = 0.18 as in the paper, Sec. VI A, or the
    E143 R1998 fit). No polarized functions.

:class:`GRSV2000Structure`
    Polarized g1 from the GRSV2000 LO "standard" grid [Glueck, Reya,
    Stratmann, Vogelsang, PRD 63 (2001) 094005] (port of ``grsv2000pdf_g1.f``,
    same CERNLIB-E104 bilinear interpolation in (ln x, ln Q2)). g1 + g2 is given
    by the Wandzura-Wilczek relation :math:`g_1+g_2 = \int_x^1 dy/y\, g_1(y)`,
    precomputed on a grid at construction (~0.1 s) and interpolated, so no
    integration is done at run time.

:class:`LHAPDFStructure`
    Leading-order parton-model F2 and g1 from any LHAPDF6 sets (optional,
    requires the ``lhapdf`` python module), WW g2 likewise gridded.

:class:`CombinedStructure` glues an unpolarized and a polarized provider
together; :func:`default_structure` returns SLAC (R = 0.18) + GRSV2000, the
input used for the paper's figures.

Target-mass corrections
-----------------------
In the strict DIS limit :math:`F_1 = F_2/(2x(1+R))`. The physics-code C++
(and hence the paper's figures) uses :math:`F_1 = (1+\gamma^2)F_2/(2x(1+R))`
with :math:`\gamma^2 = 4m^2x^2/Q^2`; this is the default here
(``target_mass=True``), pass ``target_mass=False`` for the strict limit.
"""

from __future__ import annotations

import os

import numpy as np

from .constants import M_NEUTRON, M_PROTON

__all__ = ["NucleonStructure", "SLACStructure", "GRSV2000Structure",
           "LHAPDFStructure", "CombinedStructure", "GriddedStructure",
           "default_structure", "r1998", "ww_grid"]

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _is_proton(nucleon):
    if nucleon in ("p", "proton", 1, True):
        return True
    if nucleon in ("n", "neutron", 0, False):
        return False
    raise ValueError(f"nucleon must be 'p' or 'n', got {nucleon!r}")


# ===========================================================================
# interface
# ===========================================================================
class NucleonStructure:
    """Abstract interface; see module docstring. Subclasses override what they provide."""

    #: short label used in plots/printouts
    name = "abstract"

    def F1F2(self, x, Q2, nucleon="n"):
        """Return (F1, F2) at Bjorken x and Q2 [GeV^2]."""
        raise NotImplementedError(f"{type(self).__name__} has no unpolarized SFs")

    def g1(self, x, Q2, nucleon="n"):
        """Return g1 at Bjorken x and Q2 [GeV^2]."""
        raise NotImplementedError(f"{type(self).__name__} has no polarized SFs")

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        """Return g1 + g2 (Wandzura-Wilczek unless the provider says otherwise)."""
        raise NotImplementedError(f"{type(self).__name__} has no polarized SFs")

    def g2(self, x, Q2, nucleon="n"):
        return self.g1_plus_g2(x, Q2, nucleon) - self.g1(x, Q2, nucleon)

    # --- the combinations the tagged structure functions need -------------
    #: whether F_[UU,L] keeps the target-mass factor (1+gamma^2), see FT_FL
    target_mass = False

    def FT_FL(self, x, Q2, nucleon="n"):
        r"""(F_[UU,T], F_[UU,L]) of the nucleon.

        F_[UU,T] = 2F1 and F_[UU,L] = (1+gamma^2) F2/x - 2F1 with
        gamma^2 = 4 m^2 x^2/Q^2 (Part I, inclusive limit), or F2/x - 2F1 in the
        strict DIS limit (paper Eq. (5.10)) when ``self.target_mass`` is False.
        With the SLAC provider this gives F_[UU,L] = R F_[UU,T] in both cases.
        """
        F1, F2 = self.F1F2(x, Q2, nucleon)
        x = np.asarray(x, float)
        if self.target_mass:
            mass = M_PROTON if _is_proton(nucleon) else M_NEUTRON
            tmc = 1.0 + 4.0 * mass**2 * x**2 / np.asarray(Q2, float)
        else:
            tmc = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            FL = np.where(x > 0, tmc * F2 / x - 2.0 * F1, 0.0)
        return 2.0 * F1, FL


class CombinedStructure(NucleonStructure):
    """Unpolarized SFs from ``unpolarized``, polarized ones from ``polarized``."""

    def __init__(self, unpolarized: NucleonStructure, polarized: NucleonStructure):
        self.unpolarized = unpolarized
        self.polarized = polarized
        self.name = f"{unpolarized.name}+{polarized.name}"

    def F1F2(self, x, Q2, nucleon="n"):
        return self.unpolarized.F1F2(x, Q2, nucleon)

    def FT_FL(self, x, Q2, nucleon="n"):
        return self.unpolarized.FT_FL(x, Q2, nucleon)

    def g1(self, x, Q2, nucleon="n"):
        return self.polarized.g1(x, Q2, nucleon)

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        return self.polarized.g1_plus_g2(x, Q2, nucleon)


# ===========================================================================
# R = sigma_L / sigma_T
# ===========================================================================
def r1998(x, Q2):
    """R = sigma_L/sigma_T, E143 fit [K. Abe et al., PLB 452 (1999) 194].

    Port of ``R1998`` in physics-code ``Tools/R199x.f``; valid for
    0.005 < x < 0.86, 0.5 < Q2 < 130 GeV^2.
    """
    x = np.asarray(x, float)
    Q2 = np.asarray(Q2, float)
    a = (0.0485, 0.5470, 2.0621, -0.3804, 0.5090, -0.0285)
    b = (0.0481, 0.6114, -0.3509, -0.4611, 0.7172, -0.0317)
    c = (0.0577, 0.4644, 1.8288, 12.3708, -43.1043, 41.7415)
    theta = 1.0 + 12.0 * Q2 * 0.125**2 / ((Q2 + 1.0) * (0.125**2 + x * x))
    rlog = np.log(Q2 / 0.04)
    q2thr = c[3] * x + c[4] * x**2 + c[5] * x**3
    R1 = a[0] * theta / rlog + a[1] * (1 + a[3] * x + a[4] * x * x) * x**a[5] \
        / (Q2**4 + a[2]**4) ** 0.25
    R2 = b[0] * theta / rlog + (b[1] / Q2 + b[2] / (Q2**2 + 0.3**2)) \
        * (1 + b[3] * x + b[4] * x * x) * x**b[5]
    R3 = c[0] * theta / rlog + c[1] / np.sqrt((Q2 - q2thr) ** 2 + c[2] ** 2)
    return (R1 + R2 + R3) / 3.0


# ===========================================================================
# SLAC / Bodek F2
# ===========================================================================
_BODEK_C = np.array([0., 1.0741163, 0.75531124, 3.3506491, 1.7447015,
                     3.5102405, 1.040004, 1.2299128, 0.10625394,
                     0.48132786, 1.5101467, 0.081661975, 0.65587179,
                     1.7176216, 0.12551987, 0.7473379, 1.953819,
                     0.19891522, -0.17498537, 0.0096701919, -0.035256748,
                     3.5185207, -0.59993696, 4.7615828, 0.41167589])
_BODEK_LSPIN = (0, 1, 2, 3, 2)


def _bodek(wm, qsq):
    """Bodek resonance+background function B(W, Q2) (port of NucleonStructure::bodek).

    The goto-heavy C code is rewritten with numpy masks; the logic is unchanged.
    """
    c = _BODEK_C
    pmsq, pm2, pm = 0.880324, 1.876512, 0.938256
    wm = np.asarray(wm, float)
    qsq = np.asarray(qsq, float)
    wsq = wm * wm
    omega = 1.0 + (wsq - pmsq) / qsq
    x = 1.0 / omega
    xpx = c[22] + c[23] * (x - c[24]) ** 2
    piemsq = (c[1] - pm) ** 2

    # background terms (b1, b2); the C code's "max(0,W-c)/(W-c)" is a step function
    b1 = np.where(wm > c[1], c[2], 0.0)
    eb1 = c[3] * (wm - c[1])
    # C: if eb1 > 25 skip the damping of b1 and jump straight to label1 (b2 w/o damping)
    b1 = np.where(eb1 > 25.0, b1, b1 * (1.0 - np.exp(-np.minimum(eb1, 25.0))))
    b2 = np.where(wm > c[4], 1.0 - c[2], 0.0)
    eb2 = c[5] * (wsq - c[4] ** 2)
    b2 = np.where(eb2 > 25.0, b2, b2 * (1.0 - np.exp(-np.minimum(eb2, 25.0))))
    bbkg = b1 + b2
    bres = c[2] + b2

    ressum = np.zeros_like(wm)
    nbkg = 5
    for i in range(1, 5):
        index = (i - 1) * 3 + 1 + nbkg
        ram = c[index] + (c[18] * qsq + c[19] * qsq * qsq if i == 1 else 0.0)
        rma = c[index + 1]
        if i == 3:
            rma = rma * (1 + c[20] / (1 + c[21] * qsq))
        rwd = c[index + 2]
        qstarn = np.sqrt(np.maximum(0.0, ((wsq + pmsq - piemsq) / (2 * wm)) ** 2 - pmsq))
        qstaro = np.sqrt(np.maximum(0.0, ((rma * rma - pmsq + piemsq) / (2 * rma)) ** 2 - piemsq))
        term = 6.08974 * qstarn
        termo = 6.08974 * qstaro
        j = 2 * _BODEK_LSPIN[i]
        k = j + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            gamres = rwd * (term / termo) ** k * (1 + termo**j) / (1 + term**j) / 2.0
            brwig = gamres / (((wm - rma) ** 2 + gamres**2) * 3.1415926)
            res = np.where(qstaro > 1e-10, ram * brwig / pm2, 0.0)
        ressum = ressum + res
    b = bbkg * (1.0 + (1.0 - bbkg) * xpx) + ressum * (1.0 - bres)
    return np.where(wm < 0.94, 0.0, b)


_GW_P = (0.256, 2.178, 0.898, -6.716, 3.756)
_GW_N = (0.064, 0.225, 4.106, -7.079, 3.055)


def slac_f2(x, Q2, nucleon="n"):
    """F2 of the free proton/neutron, SLAC/Bodek fit (port of f2p_b / f2n_b)."""
    proton = _is_proton(nucleon)
    mass = M_PROTON if proton else M_NEUTRON
    x = np.asarray(x, float)
    Q2 = np.asarray(Q2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.sqrt(np.maximum(Q2 * (1.0 / x - 1.0) + mass * mass, 0.0))
        nu = Q2 / (2 * mass * x)
        wwi = (2 * nu * mass + 1.642) / (Q2 + 0.376)
        t = 1.0 - 1.0 / wwi
        g = _GW_P if proton else _GW_N
        gw = g[0] * t**3 + g[1] * t**4 + g[2] * t**5 + g[3] * t**6 + g[4] * t**7
        f2 = _bodek(W, Q2) * gw * wwi * x
    return np.where((x > 1.0) | (x <= 0) | (W < mass), 0.0, f2)


class SLACStructure(NucleonStructure):
    """Unpolarized SLAC/Bodek F2 with F1 from R = sigma_L/sigma_T.

    Parameters
    ----------
    R : float or ``"r1998"``
        Constant R (default 0.18, as used for the paper's figures) or the E143 fit.
    target_mass : bool
        Include the factor (1+gamma^2), gamma^2 = 4 m^2 x^2/Q^2, in F1 (default
        True, as in the physics-code C++ that produced the paper's figures);
        False gives the strict DIS-limit relation F1 = F2/(2x(1+R)).
    """

    def __init__(self, R=0.18, target_mass=True):
        self.R = R
        self.target_mass = target_mass
        self.name = f"SLAC(R={R})"

    def F1F2(self, x, Q2, nucleon="n"):
        x = np.asarray(x, float)
        Q2 = np.asarray(Q2, float)
        F2 = slac_f2(x, Q2, nucleon)
        R = r1998(x, Q2) if isinstance(self.R, str) else self.R
        mass = M_PROTON if _is_proton(nucleon) else M_NEUTRON
        tmc = 1.0 + 4 * mass**2 * x**2 / Q2 if self.target_mass else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            F1 = np.where(x > 0, F2 * tmc / (2.0 * x * (1.0 + R)), 0.0)
        return F1, F2


# ===========================================================================
# Wandzura-Wilczek helper
# ===========================================================================
def ww_grid(g1_func, x_nodes, Q2_nodes, n_sub=40):
    r"""Tabulate :math:`\int_x^1 dy/y\, g_1(y, Q^2)` on (x_nodes, Q2_nodes).

    The integral is done in t = ln y with Gauss-Legendre on each interval
    between consecutive x nodes (n_sub points), accumulated from x = 1
    downward; ``g1_func(y, Q2)`` must be vectorized. Returns an array of shape
    (len(x_nodes), len(Q2_nodes)). x_nodes must be increasing and end at 1.
    """
    x_nodes = np.asarray(x_nodes, float)
    Q2_nodes = np.asarray(Q2_nodes, float)
    tg, wg = np.polynomial.legendre.leggauss(n_sub)
    lx = np.log(x_nodes)
    out = np.zeros((len(x_nodes), len(Q2_nodes)))
    # integral over each interval [x_i, x_{i+1}] of dy/y g1 = dt g1(e^t)
    a, b = lx[:-1], lx[1:]
    t = 0.5 * (b - a)[:, None] * tg[None, :] + 0.5 * (b + a)[:, None]   # (nint, n_sub)
    y = np.exp(t)
    for j, q2 in enumerate(Q2_nodes):
        vals = g1_func(y, np.full_like(y, q2))
        seg = 0.5 * (b - a) * np.sum(wg[None, :] * vals, axis=1)
        out[:-1, j] = np.cumsum(seg[::-1])[::-1]
    return out


def _bilinear(lx_nodes, lq_nodes, table, lx, lq):
    """Bilinear interpolation of ``table`` in (ln x, ln Q2), clamped to the grid edges."""
    lx = np.clip(lx, lx_nodes[0], lx_nodes[-1])
    lq = np.clip(lq, lq_nodes[0], lq_nodes[-1])
    i = np.clip(np.searchsorted(lx_nodes, lx) - 1, 0, len(lx_nodes) - 2)
    j = np.clip(np.searchsorted(lq_nodes, lq) - 1, 0, len(lq_nodes) - 2)
    tx = (lx - lx_nodes[i]) / (lx_nodes[i + 1] - lx_nodes[i])
    tq = (lq - lq_nodes[j]) / (lq_nodes[j + 1] - lq_nodes[j])
    return ((1 - tx) * (1 - tq) * table[i, j] + tx * (1 - tq) * table[i + 1, j]
            + (1 - tx) * tq * table[i, j + 1] + tx * tq * table[i + 1, j + 1])


# ===========================================================================
# GRSV2000 g1
# ===========================================================================
_GRSV_Q2 = np.array([0.8, 1.0, 1.25, 1.5, 2., 2.5, 4.0, 6.4, 1.0e1, 1.5e1, 2.5e1,
                     4.0e1, 6.4e1, 1.0e2, 1.8e2, 3.2e2, 5.8e2, 1.0e3, 1.8e3,
                     3.2e3, 5.8e3, 1.0e4, 1.8e4, 3.2e4, 5.8e4, 1.0e5, 1.8e5,
                     3.2e5, 5.8e5, 1.0e6])
_GRSV_X = np.array([1.e-4, 1.5e-4, 2.2e-4, 3.2e-4, 4.8e-4, 7.e-4,
                    1.e-3, 1.5e-3, 2.2e-3, 3.2e-3, 4.8e-3, 7.e-3,
                    1.e-2, 1.5e-2, 2.2e-2, 3.2e-2, 5.0e-2, 7.5e-2,
                    0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
                    0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55,
                    0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])


class GRSV2000Structure(NucleonStructure):
    """Polarized g1 (GRSV2000 LO standard) and Wandzura-Wilczek g1+g2.

    Grid range: 1e-4 <= x <= 1, 0.8 <= Q2 <= 1e6 GeV^2. Outside the range the
    Fortran code stops; here x and Q2 are **clamped to the grid edges** (with
    the (1-x)^3 factor still evaluated at the true x), which is the safe choice
    inside an MC loop. Check ``in_range(x, Q2)`` if this matters.

    Parameters
    ----------
    grid_file : str, optional
        Path to ``std2000_lo_g1.grid`` (shipped in ``taggedpd/data``).
    n_ww_x : int
        Number of log-spaced x nodes for the WW table (default 400).
    ww : {"exact", "cpp"}
        ``"exact"`` (default): accurate WW integral (agrees with adaptive
        quadrature to ~1e-4). ``"cpp"``: reproduce the 101-point left Riemann
        sum of physics-code ``NucleonStructure::getG1plusg2_grsv2000``, which
        is inaccurate at small x (-7.5% at x=0.1, factor ~3 at x=0.002); only
        meant for cross-checks against existing C++ output.
    """

    name = "GRSV2000-LO"

    def __init__(self, grid_file=None, n_ww_x=400, ww="exact"):
        if ww not in ("exact", "cpp"):
            raise ValueError("ww must be 'exact' or 'cpp'")
        self.ww = ww
        grid_file = grid_file or os.path.join(_DATA, "std2000_lo_g1.grid")
        raw = np.loadtxt(grid_file)          # (NX-1)*NQ rows, 8 columns
        nx, nq = len(_GRSV_X), len(_GRSV_Q2)
        raw = raw.reshape(nx - 1, nq, 8)     # Fortran loop: M (x) outer, N (Q2) inner
        xb = _GRSV_X[:-1, None]
        # same rescaling as the Fortran: table holds g1/(1-x)^3, zero at x=1
        self._tab = {}
        for col, key in ((6, "p"), (7, "n")):
            t = np.zeros((nx, nq))
            t[:-1] = raw[:, :, col] / (1.0 - xb) ** 3
            self._tab[key] = t
        self._lx = np.log(_GRSV_X)
        self._lq = np.log(_GRSV_Q2)

        # Wandzura-Wilczek table on a fine log-x grid
        self._ww_x = np.concatenate([np.geomspace(_GRSV_X[0], 0.9, n_ww_x - 1), [1.0]])
        self._ww_lx = np.log(self._ww_x)
        self._ww = {key: ww_grid(lambda y, q, key=key: self._g1(y, q, key),
                                 self._ww_x, _GRSV_Q2) for key in ("p", "n")}

    @staticmethod
    def in_range(x, Q2):
        x, Q2 = np.asarray(x), np.asarray(Q2)
        return (x >= _GRSV_X[0]) & (x <= 1) & (Q2 >= _GRSV_Q2[0]) & (Q2 <= _GRSV_Q2[-1])

    def _g1(self, x, Q2, key):
        x = np.asarray(x, float)
        Q2 = np.asarray(Q2, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = _bilinear(self._lx, self._lq, self._tab[key], np.log(x), np.log(Q2))
        return np.where((x > 0) & (x < 1), val * (1.0 - x) ** 3, 0.0)

    def g1(self, x, Q2, nucleon="n"):
        return self._g1(x, Q2, "p" if _is_proton(nucleon) else "n")

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        """Wandzura-Wilczek g1+g2 = int_x^1 dy/y g1(y), interpolated from the precomputed table."""
        key = "p" if _is_proton(nucleon) else "n"
        x = np.asarray(x, float)
        Q2 = np.asarray(Q2, float)
        if self.ww == "cpp":
            # sum_{i=0}^{100} g1(y_i)/y_i * step, y_i = x + i (1-x)/100  (C++ verbatim)
            step = (1.0 - x) / 100.0
            i = np.arange(101).reshape((101,) + (1,) * x.ndim)
            y = x + step * i
            return np.sum(self._g1(y, Q2, key) / y, axis=0) * step
        with np.errstate(divide="ignore", invalid="ignore"):
            val = _bilinear(self._ww_lx, self._lq, self._ww[key], np.log(x), np.log(Q2))
        return np.where((x > 0) & (x < 1), val, 0.0)


# ===========================================================================
# LHAPDF (optional)
# ===========================================================================
class LHAPDFStructure(NucleonStructure):
    r"""Leading-order parton-model structure functions from LHAPDF6 sets.

    .. math::
        F_2 = \sum_q e_q^2\, x\,(q + \bar q), \qquad
        g_1 = \tfrac12 \sum_q e_q^2\, (\Delta q + \Delta\bar q),

    with u <-> d interchanged for the neutron (isospin symmetry), and
    F1 = F2/(2x(1+R)). g1+g2 from the Wandzura-Wilczek relation, tabulated at
    construction on (x, Q2) nodes and interpolated. Requires the ``lhapdf``
    python bindings; either set may be ``None``.

    LHAPDF is called point by point here; for event generation wrap it as
    ``GriddedStructure(LHAPDFStructure(...))``. (Untested in this repository:
    the lhapdf module was not available when it was written.)

    Parameters
    ----------
    unpolarized_set, polarized_set : str or None
        LHAPDF set names, e.g. ``"CT18LO"``, ``"NNPDFpol11_100"``.
    member : int
        Set member (0 = central).
    R : float or "r1998"
    nf : int
        Number of light flavours in the sums (default 3 for g1, 5 for F2 is
        set via ``nf_unpol``).
    """

    def __init__(self, unpolarized_set=None, polarized_set=None, member=0,
                 R=0.18, nf_unpol=5, nf_pol=3, ww_Q2_nodes=None, n_ww_x=300):
        try:
            import lhapdf  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LHAPDFStructure needs the 'lhapdf' python module") from exc
        import lhapdf
        lhapdf.setVerbosity(0)
        self._u = lhapdf.mkPDF(unpolarized_set, member) if unpolarized_set else None
        self._p = lhapdf.mkPDF(polarized_set, member) if polarized_set else None
        self.R = R
        self.nf_unpol, self.nf_pol = nf_unpol, nf_pol
        self.name = f"LHAPDF({unpolarized_set},{polarized_set})"
        if self._p is not None:
            q2min = max(self._p.q2Min, 1.0)
            q2max = min(self._p.q2Max, 1e5)
            self._ww_q2 = (np.geomspace(q2min, q2max, 30) if ww_Q2_nodes is None
                           else np.asarray(ww_Q2_nodes, float))
            xmin = max(self._p.xMin, 1e-5)
            self._ww_x = np.concatenate([np.geomspace(xmin, 0.95, n_ww_x - 1), [1.0]])
            self._ww = {k: ww_grid(lambda y, q, k=k: self.g1(y, q, k), self._ww_x, self._ww_q2)
                        for k in ("p", "n")}

    # e_q^2 for d, u, s, c, b  (PDG ids 1..5)
    _E2 = {1: 1 / 9, 2: 4 / 9, 3: 1 / 9, 4: 4 / 9, 5: 1 / 9}

    def _sum(self, pdf, x, Q2, nucleon, nf):
        proton = _is_proton(nucleon)
        x = np.atleast_1d(np.asarray(x, float))
        Q2 = np.broadcast_to(np.asarray(Q2, float), x.shape)
        out = np.zeros(x.shape)
        for idx in np.ndindex(x.shape):
            xi, qi = float(x[idx]), float(Q2[idx])
            if not (0 < xi < 1):
                continue
            s = 0.0
            for f in range(1, nf + 1):
                fl = f
                if not proton and f in (1, 2):   # isospin: u <-> d
                    fl = 3 - f
                s += self._E2[f] * (pdf.xfxQ2(fl, xi, qi) + pdf.xfxQ2(-fl, xi, qi))
            out[idx] = s
        return out   # = sum e^2 x (q + qbar)

    def F1F2(self, x, Q2, nucleon="n"):
        if self._u is None:
            raise NotImplementedError("no unpolarized set given")
        F2 = self._sum(self._u, x, Q2, nucleon, self.nf_unpol)
        x = np.asarray(x, float)
        R = r1998(x, Q2) if isinstance(self.R, str) else self.R
        return F2 / (2 * x * (1 + R)), F2

    def g1(self, x, Q2, nucleon="n"):
        if self._p is None:
            raise NotImplementedError("no polarized set given")
        return 0.5 * self._sum(self._p, x, Q2, nucleon, self.nf_pol) / np.asarray(x, float)

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        key = "p" if _is_proton(nucleon) else "n"
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = _bilinear(np.log(self._ww_x), np.log(self._ww_q2), self._ww[key],
                            np.log(x), np.log(np.asarray(Q2, float)))
        return np.where((x > 0) & (x < 1), val, 0.0)


class GriddedStructure(NucleonStructure):
    """Tabulate any provider once on a (ln x, ln Q2) grid and interpolate bilinearly.

    Use this to make slow providers (e.g. :class:`LHAPDFStructure`, which calls
    LHAPDF point by point) fast enough for event generation. Tabulated: F1, F2
    (if available), g1, g1+g2 (if available), for both nucleons. Outside the grid
    the values are clamped to the edges; at x >= 1 everything is 0.

    Parameters
    ----------
    provider : NucleonStructure
    x_nodes, Q2_nodes : array_like, optional
        Default: 300 log-spaced x in [1e-4, 0.999] and 60 log-spaced Q2 in [1, 1e4].
    """

    def __init__(self, provider: NucleonStructure, x_nodes=None, Q2_nodes=None):
        self.provider = provider
        self.name = f"grid({provider.name})"
        self.target_mass = getattr(provider, "target_mass", False)
        if isinstance(provider, CombinedStructure):
            self.target_mass = getattr(provider.unpolarized, "target_mass", False)
        self._x = np.geomspace(1e-4, 0.999, 300) if x_nodes is None else np.asarray(x_nodes, float)
        self._q = np.geomspace(1.0, 1e4, 60) if Q2_nodes is None else np.asarray(Q2_nodes, float)
        self._lx, self._lq = np.log(self._x), np.log(self._q)
        X, Q = np.meshgrid(self._x, self._q, indexing="ij")
        self._tab = {}
        for nuc in ("p", "n"):
            for key, fn in (("F", lambda: provider.F1F2(X, Q, nuc)),
                            ("g1", lambda: provider.g1(X, Q, nuc)),
                            ("gg", lambda: provider.g1_plus_g2(X, Q, nuc))):
                try:
                    self._tab[(key, nuc)] = fn()
                except NotImplementedError:
                    pass

    def _interp(self, table, x, Q2):
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = _bilinear(self._lx, self._lq, table, np.log(x), np.log(np.asarray(Q2, float)))
        return np.where((x > 0) & (x < 1), val, 0.0)

    def _get(self, key, nucleon):
        k = (key, "p" if _is_proton(nucleon) else "n")
        if k not in self._tab:
            raise NotImplementedError(f"{self.provider.name} provides no {key}")
        return self._tab[k]

    def F1F2(self, x, Q2, nucleon="n"):
        F1, F2 = self._get("F", nucleon)
        return self._interp(F1, x, Q2), self._interp(F2, x, Q2)

    def g1(self, x, Q2, nucleon="n"):
        return self._interp(self._get("g1", nucleon), x, Q2)

    def g1_plus_g2(self, x, Q2, nucleon="n"):
        return self._interp(self._get("gg", nucleon), x, Q2)


def default_structure(R=0.18, target_mass=True, ww="exact"):
    """SLAC/Bodek F2 (R = 0.18) + GRSV2000 LO g1 with WW g2: the paper's numerical input."""
    return CombinedStructure(SLACStructure(R=R, target_mass=target_mass),
                             GRSV2000Structure(ww=ww))
