r"""Nonrelativistic deuteron radial wave functions (Yukawa parametrizations).

This is a port of ``TYukawaPWF`` / ``TDeuteron::Wavefunction`` from nucqcd.
The radial wave functions are parametrized as a discrete sum of Yukawa terms
[Lacombe et al., PLB 101 (1981) 139]:

.. math::

    u(r) = \sum_i C_i e^{-m_i r}, \qquad
    w(r) = \sum_i D_i e^{-m_i r}\left(1 + \frac{3}{m_i r} + \frac{3}{(m_i r)^2}\right),

which in momentum space become

.. math::

    U(k) = \sqrt{\tfrac{2}{\pi}} \sum_i \frac{C_i}{k^2 + m_i^2}, \qquad
    W(k) = \sqrt{\tfrac{2}{\pi}} \sum_i \frac{-D_i}{k^2 + m_i^2}.

The minus sign in W comes from the factor i^{-L} = -1 of the L=2 Fourier--Bessel
transform; it is part of the C++ ``GetWp`` as well.

The boundary conditions (u(0)=0, regular w at r->0) fix the last C and the last
three D coefficients:
:math:`\sum C_i = \sum D_i = \sum D_i m_i^2 = \sum D_i/m_i^2 = 0`.
They are imposed here exactly as in the C++ constructor (the tabulated last
coefficients are overwritten).

Normalization and sign conventions
----------------------------------
* ``U``/``W`` are normalized as :math:`\int_0^\infty dk\,k^2 [U^2 + W^2] = 1`
  (standard nonrelativistic normalization, arXiv:2603.23700 Eq. (3.25) with
  :math:`f_{L,\rm nr} = U_L/\sqrt{4\pi}`).
* With Yukawa terms the norm integrals are known in closed form,
  :math:`\int_0^\infty dk\,k^2 U^2 = \sum_{ij} C_i C_j/(m_i+m_j)`, which gives an
  *exact* reference for the normalization tests (see ``norm_S`` / ``norm_D``).
* **D-wave sign.** arXiv:2603.23700's formulas (e.g. the node of
  :math:`f_0 - f_2/\sqrt{2}` at k ~ 0.3 GeV, the extremal tensor asymmetry at
  :math:`f_2/f_0=\sqrt{2}`) assume a **positive** D wave in momentum space,
  i.e. :math:`W(k) > 0` at small k (where :math:`W \propto k^2`). This is
  enforced explicitly: the overall sign of W is chosen such that
  :math:`\lim_{k\to0} W(k)/k^2 > 0`, independent of the sign convention of the
  tabulated D coefficients. For the three parametrizations shipped here this
  agrees with the C++ ``GetWp`` (the factor ``d_wave_sign`` is +1), which is
  verified in the test-suite.
* Light-front radial functions (arXiv:2603.23700 Eq. (3.24)):
  :math:`f_L(k) = \sqrt{E(k)}\, f_{L,\rm nr}(k) = \sqrt{E(k)/(4\pi)}\, U_L(k)`,
  normalized as :math:`4\pi\int dk\, k^2/E\,[f_0^2+f_2^2] = 1` (Eq. (3.23)).

All momenta in GeV; U, W in GeV^{-3/2}; f0, f2 in GeV^{-1}.
"""

from __future__ import annotations

import numpy as np

from .constants import HBARC, M_NUCLEON

__all__ = ["DeuteronWavefunction", "available_wavefunctions"]


# ---------------------------------------------------------------------------
# Parameter tables, copied verbatim from nucqcd Tools/TDeuteron.cpp.
# C, D in fm^{-1/2}; masses in fm^{-1}.
# ---------------------------------------------------------------------------

_SQ2PI = np.sqrt(2.0 / np.pi)

_PARAMS = {
    # CD-Bonn, R. Machleidt, PRC 63 (2001) 024001.  m_i = alpha + (i-1) m0
    "CDBonn": dict(
        c=[+.88472985e+0, -.26408759e+0, -.44114404e-1, -.14397512e+2,
           +.85591256e+2, -.31876761e+3, +.70336701e+3, -.90049586e+3,
           +.66145441e+3, -.25958894e+3, 0.0],
        d=[+.22623762e-1, -.50471056e+0, +.56278897e+0, -.16079764e+2,
           +.11126803e+3, -.44667490e+3, +.10985907e+4, -.16114995e+4,
           0.0, 0.0, 0.0],
        alpha=0.2315380, m0=0.9),
    # Paris, Lacombe et al., PLB 101 (1981) 139.
    "Paris": dict(
        c=[+.88688076e-0, -.34717093e+0, -.30502380e+1, +.56207766e+2,
           -.74957334e+3, +.53365279e+4, -.22706863e+5, +.60434469e+5,
           -.10292058e+6, +.11223357e+6, -.75925226e+5, +.29059715e+5, 0.0],
        d=[+.23135193e-1, -.85604572e+0, +.56068193e+1, -.69462922e+2,
           +.41631118e+3, -.12546621e+4, +.12387830e+4, +.33739172e+4,
           -.13041151e+5, +.19512524e+5, 0.0, 0.0, 0.0],
        alpha=0.23162461, m0=1.0),
    # Argonne v18, Wiringa, Stoks, Schiavilla, PRC 51 (1995) 38.
    # The tabulated values are momentum-space coefficients and are divided
    # by sqrt(2/pi) to convert them to the r-space convention used here.
    "AV18": dict(
        c=[v / _SQ2PI for v in (
            0.706699E+00, -0.169743E+00, 0.112368E+01, -0.852995E+01,
            0.195033E+02, -0.757831E+02, 0.283739E+03, -0.694734E+03,
            0.885257E+03, -0.720739E+03, 0.412969E+03, -0.103336E+03)],
        d=[v / _SQ2PI for v in (
            0.176655E-01, -0.124551E+00, -0.108815E+01, 0.384848E+01,
            -0.852442E+01, 0.209435E+02, -0.490728E+02, 0.577382E+02,
            -0.127114E+01, -0.628361E+02, 0.581016E+02, -0.177062E+02)],
        masses=[0.2316, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 9.5, 11.0]),
}

#: case-insensitive aliases
_ALIASES = {"av18": "AV18", "cdbonn": "CDBonn", "cd-bonn": "CDBonn",
            "paris": "Paris"}


def available_wavefunctions():
    """Names of the shipped deuteron wave function parametrizations."""
    return list(_PARAMS)


class DeuteronWavefunction:
    """Deuteron S- and D-wave radial functions from a Yukawa parametrization.

    Parameters
    ----------
    name : str
        ``"AV18"``, ``"CDBonn"`` or ``"Paris"`` (case-insensitive).

    Notes
    -----
    Evaluation is a short vectorized sum over ~12 poles; it accepts numpy
    arrays of any shape and is fast enough to be called inside an MC loop.
    """

    def __init__(self, name: str = "AV18"):
        key = _ALIASES.get(name.lower(), name)
        if key not in _PARAMS:
            raise ValueError(f"unknown wave function {name!r}; "
                             f"choose from {available_wavefunctions()}")
        self.name = key
        p = _PARAMS[key]
        c = np.array(p["c"], dtype=float)
        d = np.array(p["d"], dtype=float)
        n = len(c)
        if "masses" in p:
            m = np.array(p["masses"], dtype=float)
        else:
            m = p["alpha"] + np.arange(n) * p["m0"]

        # --- boundary conditions (same as the TYukawaPWF constructor) -----
        # C_n = -sum_{i<n} C_i   ->  u(r=0) = 0
        c[-1] = -np.sum(c[:-1])
        # last three D from: sum D/m^2 = 0, sum D = 0, sum D m^2 = 0
        dfix = d[:-3]
        mfix = m[:-3]
        rhs = -np.array([np.sum(dfix / mfix**2), np.sum(dfix),
                         np.sum(dfix * mfix**2)])
        mlast = m[-3:]
        mat = np.vstack([1.0 / mlast**2, np.ones(3), mlast**2])
        d[-3:] = np.linalg.solve(mat, rhs)

        self._c = c           # [fm^-1/2]
        self._d = d           # [fm^-1/2]
        self._m = m           # [fm^-1]

        # Enforce positive D wave at small k:  W(k) -> sqrt(2/pi) k^2 sum D/m^4
        # (with k in fm^-1).  d_wave_sign = +1 means the C++ GetWp sign is kept.
        small_k_coeff = np.sum(self._d / self._m**4)
        self.d_wave_sign = 1.0 if small_k_coeff > 0 else -1.0

    # ------------------------------------------------------------------
    # nonrelativistic momentum-space radial functions
    # ------------------------------------------------------------------
    def _yukawa_sum(self, coeff, k):
        """sqrt(2/pi) sum_i coeff_i/(q^2+m_i^2), converted to GeV^-3/2."""
        q = np.asarray(k, dtype=float)[..., None] / HBARC  # GeV -> fm^-1
        s = np.sum(coeff / (q**2 + self._m**2), axis=-1)
        return _SQ2PI * s / HBARC**1.5

    def U(self, k):
        """S-wave U(k) [GeV^-3/2], k = |k| in GeV. Norm: int dk k^2 (U^2+W^2) = 1."""
        return self._yukawa_sum(self._c, k)

    def W(self, k):
        """D-wave W(k) [GeV^-3/2] with the positive sign convention (see module doc)."""
        return self.d_wave_sign * self._yukawa_sum(-self._d, k)

    # ------------------------------------------------------------------
    # light-front radial functions, arXiv:2603.23700 Eq. (3.24)
    # ------------------------------------------------------------------
    @staticmethod
    def energy(k):
        """On-shell nucleon energy E(k) = sqrt(k^2 + m^2) [GeV], Eq. (3.19b)."""
        return np.sqrt(np.asarray(k, dtype=float) ** 2 + M_NUCLEON**2)

    def f0(self, k):
        """LF S-wave radial function f_0(k) = sqrt(E/4pi) U(k) [GeV^-1], Eq. (3.24)."""
        return np.sqrt(self.energy(k) / (4.0 * np.pi)) * self.U(k)

    def f2(self, k):
        """LF D-wave radial function f_2(k) = sqrt(E/4pi) W(k) [GeV^-1], Eq. (3.24)."""
        return np.sqrt(self.energy(k) / (4.0 * np.pi)) * self.W(k)

    def f0f2(self, k):
        """Both LF radial functions at once (saves one sqrt)."""
        pref = np.sqrt(self.energy(k) / (4.0 * np.pi))
        return pref * self.U(k), pref * self.W(k)

    # ------------------------------------------------------------------
    # exact (closed-form) normalization integrals
    # ------------------------------------------------------------------
    def _pair_integral(self, a, b):
        # int_0^inf dk k^2 [sqrt(2/pi) sum a_i/(k^2+m_i^2)][sqrt(2/pi) sum b_j/(k^2+m_j^2)]
        #  = (2/pi) sum_ij a_i b_j * pi/(2(m_i+m_j)) = sum_ij a_i b_j/(m_i+m_j)
        # The coefficients of e.g. the Paris set reach 1e5 and the double sum
        # cancels from O(1e10) terms down to O(1): float64 products alone lose
        # ~1e-6. The sum is therefore done in exact rational arithmetic on the
        # (float) coefficients -- it is only a 13x13 sum.
        from fractions import Fraction
        A = [Fraction(float(v)) for v in a]
        B = [Fraction(float(v)) for v in b]
        M = [Fraction(float(v)) for v in self._m]
        n = len(M)
        return float(sum(A[i] * B[j] / (M[i] + M[j]) for i in range(n) for j in range(n)))

    @property
    def norm_S(self):
        """Exact S-wave norm  int dk k^2 U^2  (dimensionless)."""
        return self._pair_integral(self._c, self._c)

    @property
    def norm_D(self):
        """Exact D-state probability  int dk k^2 W^2 = omega_2 (arXiv:2603.23700 Eq. (A7))."""
        return self._pair_integral(self._d, self._d)

    @property
    def norm(self):
        """Exact total norm  int dk k^2 (U^2 + W^2); equals Eq. (3.23) exactly."""
        return self.norm_S + self.norm_D

    @property
    def residue_coefficient(self):
        """Leading-pole coefficient C_1 [fm^-1/2] (Bethe--Peierls pole at k^2 = -m_1^2)."""
        return float(self._c[0])

    def __repr__(self):
        return (f"DeuteronWavefunction({self.name!r}: norm={self.norm:.6f}, "
                f"P_D={self.norm_D:.4f})")
