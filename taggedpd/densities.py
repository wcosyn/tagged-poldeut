r"""Light-front momentum distributions of the active neutron in the deuteron.

Implements Sec. IV of W. Cosyn and C. Weiss, Phys. Rev. C 114, 025202 (2026),
arXiv:2603.23700; equation numbers below refer to it.

Variables
---------
The tagged (spectator) proton is described by its LF momentum fraction
``alpha_p`` (Eq. (2.12), :math:`0<\alpha_p<2`) and the modulus of its transverse
momentum ``pT`` = :math:`|\mathbf p_{pT}|` [GeV]. They map onto the pn c.m.
relative momentum **k** via Eq. (3.20):

.. math::
    E = \sqrt{\frac{m^2 + p_T^2}{\alpha_p(2-\alpha_p)}}, \quad
    k^z = E(\alpha_p - 1), \quad k_T = p_T, \quad
    \cos\theta_k = k^z/|k|, \ \sin\theta_k = k_T/|k|.

The transverse basis is oriented along the proton transverse momentum
(:math:`e_{x'} = p_{pT}/|p_{pT}|`, Eq. (2.25)), so the azimuth of **k** is zero
and every distribution depends on (alpha_p, pT) only.

Normalized distributions (all in GeV^{-2}, include the flux factor 1/(2-alpha_p))
--------------------------------------------------------------------------------
=================  ==========================================  ===========
attribute          meaning                                     eq.
=================  ==========================================  ===========
``UU``             unpolarized n in unpolarized D              (4.18)
``TLL_U``          unpolarized n, T_LL tensor-polarized D      (4.19)
``TLT_U``          unpolarized n, T_LT tensor-polarized D      (4.19)
``TTT_U``          unpolarized n, T_TT tensor-polarized D      (4.19)
``SL_SL``          long. pol. n in long. pol. D  (favored)     (4.30)
``ST_SL``          long. pol. n in transv. pol. D (unfavored)  (4.31)
``ST_ST_par``      transv. pol. n (x') in transv. pol. D (x')  (4.36a)
``ST_ST_perp``     transv. pol. n (y') in transv. pol. D (y')  (4.36b)
``SL_ST``          transv. pol. n (x') in long. pol. D         (4.38a)
=================  ==========================================  ===========

The tensor distributions are those of the *normalized* spherical tensors
e_LL, e_LT, e_TT (arXiv:2603.23700 Eqs. (4.19)-(4.20)); the factors sqrt(3/2), sqrt(2),
1/sqrt(2) of Eq. (2.27) are applied where the general-polarization
distribution is assembled (Eq. (4.22)) and in the structure functions.

.. note::
   The C++ ``Poldeut::getLFdistributions_new`` returns tensor distributions
   that differ by those factors: ``P_TLL[C++] = sqrt(3/2) TLL_U``,
   ``P_TLT[C++] = sqrt(2) TLT_U``, ``P_TTT[C++] = TTT_U/sqrt(2)`` (see the
   regression test in ``tests/test_densities.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import M_NUCLEON
from .wavefunctions import DeuteronWavefunction

__all__ = ["cm_momentum", "LFDistributions", "NeutronDistributions"]

SQRT2 = np.sqrt(2.0)
SQRT6 = np.sqrt(6.0)


def cm_momentum(alpha_p, pT, m=M_NUCLEON):
    r"""Relative pn c.m. momentum from the spectator LF variables, Eq. (3.20).

    Parameters
    ----------
    alpha_p : array_like
        Spectator LF momentum fraction, 0 < alpha_p < 2.
    pT : array_like
        Modulus of the spectator transverse momentum [GeV].

    Returns
    -------
    E, kz, kT, k : ndarray
        E(k) = M_pn/2 [GeV], longitudinal and transverse components and modulus
        of **k** [GeV]. Note k_T = +p_pT (Eq. (3.20a)).
    """
    alpha_p = np.asarray(alpha_p, dtype=float)
    pT = np.asarray(pT, dtype=float)
    E = np.sqrt((m * m + pT * pT) / (alpha_p * (2.0 - alpha_p)))
    kz = E * (alpha_p - 1.0)
    kT = pT
    k = np.sqrt(kz * kz + kT * kT)
    return E, kz, kT, k


def spectator_lf_from_rest_frame(p_vec, M_D=None, m=M_NUCLEON):
    r"""(alpha_p, p_pT) from the proton 3-momentum in the deuteron rest frame, Eq. (2.13).

    ``p_vec[..., 2]`` is the component along the collinear z axis (opposite to
    the virtual photon momentum **q**). Useful for MC generators that sample the
    spectator in the rest frame.
    """
    from .constants import M_DEUTERON
    M_D = M_DEUTERON if M_D is None else M_D
    p_vec = np.asarray(p_vec, dtype=float)
    E = np.sqrt(np.sum(p_vec**2, axis=-1) + m * m)
    alpha_p = 2.0 * (E + p_vec[..., 2]) / M_D
    pT = np.hypot(p_vec[..., 0], p_vec[..., 1])
    return alpha_p, pT


@dataclass
class NeutronDistributions:
    """Container for the normalized LF neutron distributions [GeV^-2].

    Attribute names follow arXiv:2603.23700's labels ``P_[deuteron, neutron]``
    (see the table in the module docstring). Arrays have the broadcast shape of
    the (alpha_p, pT) input.
    """
    UU: np.ndarray
    TLL_U: np.ndarray
    TLT_U: np.ndarray
    TTT_U: np.ndarray
    SL_SL: np.ndarray
    ST_SL: np.ndarray
    ST_ST_par: np.ndarray
    ST_ST_perp: np.ndarray
    SL_ST: np.ndarray

    # ---- probabilistic distributions, Sec. IV E ------------------------------
    # Pure deuteron states with spin +1 along e_z (L+) or along e_x' (T+).
    # Unpolarized part for these states (Eqs. (4.45), (4.47)):
    @property
    def _unpol_Lplus(self):
        # P[U,U] + (1/sqrt6) P[TLL,U]   (T_D = e_LL/sqrt6 for Lambda=+1 along z)
        return self.UU + self.TLL_U / SQRT6

    @property
    def _unpol_Tplus(self):
        # P[U,U] - 1/(2 sqrt6) P[TLL,U] + 1/(2 sqrt2) P[TTT,U]   (Eq. (2.32b), phi_N = phi_p)
        return self.UU - self.TLL_U / (2 * SQRT6) + self.TTT_U / (2 * SQRT2)

    def P_L_L(self, sign=+1):
        """P[L+, L+/-]: neutron long. spin +/-1/2 in deuteron long. spin +1, Eq. (4.45a)."""
        return 0.5 * (self._unpol_Lplus + sign * self.SL_SL)

    def P_T_L(self, sign=+1):
        """P[T+, L+/-]: neutron long. spin +/-1/2 in deuteron transverse (x') spin +1, Eq. (4.45b)."""
        return 0.5 * (self._unpol_Tplus + sign * self.ST_SL)

    def P_T_T(self, sign=+1):
        """P[T+, T+/-]: neutron transv. (x') spin +/-1/2 in deuteron transv. (x') spin +1, Eq. (4.47a)."""
        return 0.5 * (self._unpol_Tplus + sign * self.ST_ST_par)

    def P_L_T(self, sign=+1):
        """P[L+, T+/-]: neutron transv. (x') spin +/-1/2 in deuteron long. spin +1, Eq. (4.47b)."""
        return 0.5 * (self._unpol_Lplus + sign * self.SL_ST)

    # ratios, Eqs. (4.48)-(4.49): probability that the neutron has spin +1/2
    @property
    def R_LL(self):
        """R[L+,L+], Eq. (4.48a)."""
        return self.P_L_L(+1) / self._unpol_Lplus

    @property
    def R_TL(self):
        """R[T+,L+], Eq. (4.48b)."""
        return self.P_T_L(+1) / self._unpol_Tplus

    @property
    def R_TT(self):
        """R[T+,T+], Eq. (4.49a)."""
        return self.P_T_T(+1) / self._unpol_Tplus

    @property
    def R_LT(self):
        """R[L+,T+], Eq. (4.49b)."""
        return self.P_L_T(+1) / self._unpol_Lplus

    # ---- general polarization (Eqs. (4.22), (4.39)) --------------------------
    def unpolarized(self, TLL=0.0, TLT=0.0, TTT=0.0, dphi_TL=0.0, dphi_TT=0.0):
        r"""P[U](alpha_p, pT | T_D) for general tensor polarization, Eq. (4.22).

        ``dphi_TL`` = phi_p - phi_{T_L}, ``dphi_TT`` = phi_p - phi_{T_T}.
        """
        return (self.UU + np.sqrt(1.5) * TLL * self.TLL_U
                + SQRT2 * TLT * np.cos(dphi_TL) * self.TLT_U
                + TTT / SQRT2 * np.cos(2 * dphi_TT) * self.TTT_U)


class LFDistributions:
    """Evaluates the LF neutron distributions for a given deuteron wave function.

    Parameters
    ----------
    wavefunction : str or DeuteronWavefunction
        ``"AV18"`` (default), ``"CDBonn"``, ``"Paris"`` or an instance.

    Examples
    --------
    >>> lf = LFDistributions("AV18")
    >>> P = lf(alpha_p=1.0, pT=0.1)
    >>> P.UU, P.SL_SL
    """

    def __init__(self, wavefunction="AV18", m=M_NUCLEON):
        if isinstance(wavefunction, str):
            wavefunction = DeuteronWavefunction(wavefunction)
        self.wf = wavefunction
        self.m = m

    def __call__(self, alpha_p, pT) -> NeutronDistributions:
        return self.compute(alpha_p, pT)

    def compute(self, alpha_p, pT) -> NeutronDistributions:
        """All normalized distributions at (alpha_p, pT); vectorized, see class doc."""
        m = self.m
        alpha_p, pT = np.broadcast_arrays(np.asarray(alpha_p, float),
                                          np.asarray(pT, float))
        E, kz, kT, k = cm_momentum(alpha_p, pT, m)
        f0, f2 = self.wf.f0f2(k)
        flux = 1.0 / (2.0 - alpha_p)          # LF flux factor, Eq. (4.14)

        # angular variables; at k = 0 the direction is undefined but every
        # angular factor multiplies f2 (or kT), so any finite value is fine.
        k_safe = np.where(k > 0, k, 1.0)
        cos_t = np.where(k > 0, kz / k_safe, 1.0)
        sin_t = np.where(k > 0, kT / k_safe, 0.0)

        # --- unpolarized & tensor-polarized, Eqs. (4.18)-(4.20) --------------
        UU = flux * (f0 * f0 + f2 * f2)
        tens = -3.0 * flux * (2.0 * f0 + f2 / SQRT2) * f2 / SQRT2
        TLL_U = tens * (3.0 * cos_t**2 - 1.0) / SQRT6    # k.e_LL.k/|k|^2
        TLT_U = tens * SQRT2 * cos_t * sin_t             # k.e_LT.k/|k|^2
        TTT_U = tens * sin_t**2 / SQRT2                  # k.e_TT.k/|k|^2

        # --- polarized: Melosh-rotation coefficients, Eqs. (4.30)-(4.31) ----
        # common factor (E + k^z) / ((m^2 + kT^2) |k|^2); finite as k->0
        # because it always multiplies kT (kT <= |k|).
        common = (E + kz) / ((m * m + kT * kT) * k_safe**2)
        A_eq = 1.0 - E * kT**2 * common                       # (4.30b)
        B_eq = m * kT**2 * common                             # (4.30c)
        A_ne = (-k**2 + E * kz) * kT * common                 # (4.31b)
        B_ne = m * kz * kT * common                           # (4.31c)

        fm = f0 - f2 / SQRT2          # (f0 - f2/sqrt2): node at k ~ 0.3 GeV
        fp = f0 + SQRT2 * f2          # (f0 + sqrt2 f2)

        SL_SL = flux * fm * (A_eq * fm + B_eq * fp)          # (4.30a)
        ST_SL = flux * fm * (A_ne * fm - B_ne * fp)          # (4.31a)
        ST_ST_par = flux * fm * (A_eq * fp + B_eq * fm)      # (4.36a)
        ST_ST_perp = flux * fm * fp                          # (4.36b)
        SL_ST = flux * fm * (-A_ne * fp + B_ne * fm)         # (4.38a)

        return NeutronDistributions(UU, TLL_U, TLT_U, TTT_U, SL_SL, ST_SL,
                                    ST_ST_par, ST_ST_perp, SL_ST)

    # convenience: the c.m.-frame density N(k) = (2 - alpha_p) P, Eq. (4.14)
    def N(self, alpha_p, pT):
        """Same distributions with the flux factor stripped, N = (2-alpha_p) P, Eq. (4.14)."""
        P = self.compute(alpha_p, pT)
        fac = 2.0 - np.asarray(alpha_p, float)
        return NeutronDistributions(*(getattr(P, f) * fac for f in
                                      NeutronDistributions.__dataclass_fields__))
