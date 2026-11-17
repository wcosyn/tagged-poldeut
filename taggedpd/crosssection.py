r"""Tagged DIS on the polarized deuteron: structure functions, cross section, asymmetries.

Process (paper Eq. (2.1)), here with a tagged proton and an active neutron
(``tagged="n"`` swaps the roles):

.. math::  e(l, \lambda_e) + D(p_D | S_D, T_D) \to e'(l') + X + p(p_p)

Variables
---------
x        : rescaled Bjorken variable x = 2 x_D = Q^2/(p_D.q), 0 < x < 2  (Eq. (2.4))
Q2       : [GeV^2]
y        : p_D.q / p_D.l ; from the e-D invariant s: y = 2 Q^2 / (x (s - M_D^2))
alpha_p  : spectator LF momentum fraction (Eq. (2.12)), kinematic limit alpha_p < 2 - x
pT       : |p_pT| [GeV]
phi_p    : spectator azimuth relative to the lepton plane (Part I, Fig. 1) [rad]
lambda_e : electron helicity, +-1/2 (enters the master formula as 2 lambda_e)

Cross section
-------------
Part I master formula (Eqs. (4.36)-(4.39) of Part I = Eq. (2.18) of the paper):

.. math::
    d\sigma = \frac{2\pi\alpha^2 y^2}{Q^4(1-\epsilon)} \frac{dx}{2} dQ^2
              \frac{d\psi_{l'}}{2\pi} (\mathcal F_U + \mathcal F_S + \mathcal F_T)\, d\Gamma_p,
    \qquad d\Gamma_p = \frac{1}{2(2\pi)^3}\frac{d\alpha_p}{\alpha_p} d^2p_{pT}

and the reduced cross section (Eq. (2.21b)) :math:`\sigma_{\rm red,D} = \tfrac{x}{2}(\mathcal F_U+\mathcal F_S+\mathcal F_T)`,
normalized such that :math:`\int d\Gamma_p\,\sigma_{\rm red,D} \approx F_{2n} - (1-\epsilon)F_{Ln}`.

Structure functions in the impulse approximation: paper Eqs. (5.12), (5.14)-(5.16),
(5.26)-(5.27). The T-odd structures (and all others listed in Eqs. (5.17), (5.28)) are
zero in the IA. Leading-twist and power-suppressed structures are all kept, exactly as
written in the paper (the power-suppressed ones are only rough estimates, Sec. III D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ALPHA_EM, GEV2_TO_NB, M_DEUTERON, M_NUCLEON, TWO_2PI3
from .densities import LFDistributions
from .nucleon import NucleonStructure, default_structure

__all__ = ["Polarization", "Kinematics", "TaggedDIS"]

SQRT2 = np.sqrt(2.0)
SQRT32 = np.sqrt(1.5)


# ===========================================================================
# kinematics helpers
# ===========================================================================
@dataclass
class Kinematics:
    r"""Inclusive (electron-side) kinematics in the deuteron collinear frame.

    Build with :meth:`from_s`, :meth:`from_y` or :meth:`from_epsilon`.

    Attributes
    ----------
    x, Q2, y, epsilon, gamma
        gamma = x M_D / Q (= 2 x_D M_D / Q, Part I Eq. (2.9)); epsilon from Part I
        Eq. (epsilon_from_y). With ``dis_limit=True``, gamma is set to 0 in epsilon
        and in the target-polarization axis angles (as in the paper's Sec. VI).
    """
    x: np.ndarray
    Q2: np.ndarray
    y: np.ndarray
    epsilon: np.ndarray
    gamma: np.ndarray

    @staticmethod
    def _gamma(x, Q2):
        return np.asarray(x, float) * M_DEUTERON / np.sqrt(np.asarray(Q2, float))

    @staticmethod
    def epsilon_from_y(y, gamma):
        g2y2 = gamma**2 * y**2 / 4.0
        return (1.0 - y - g2y2) / (1.0 - y + y * y / 2.0 + g2y2)

    @classmethod
    def from_y(cls, x, Q2, y, dis_limit=False):
        x, Q2, y = (np.asarray(v, float) for v in (x, Q2, y))
        gamma = 0.0 * x if dis_limit else cls._gamma(x, Q2)
        return cls(x, Q2, y, cls.epsilon_from_y(y, gamma), gamma)

    @classmethod
    def from_s(cls, x, Q2, s, dis_limit=False, per_nucleon=False):
        r"""Kinematics from the invariant energy squared [GeV^2] (massless electron).

        ``per_nucleon=False``: s = (l + p_D)^2 (electron-deuteron),
        y = 2 Q^2 / (x (s - M_D^2)).
        ``per_nucleon=True``: s = (l + p_D/2)^2 (electron-nucleon, e.g. 4 E_e E_A/A
        for colliding beams; the convention of the paper's Figs. 9 and 15,
        "s = 800 GeV^2" for 5 x 41 GeV), y = Q^2 / (x (s - M_D^2/4)).
        """
        x, Q2, s = (np.asarray(v, float) for v in (x, Q2, s))
        if per_nucleon:
            y = Q2 / (x * (s - M_DEUTERON**2 / 4.0))
        else:
            y = 2.0 * Q2 / (x * (s - M_DEUTERON**2))
        return cls.from_y(x, Q2, y, dis_limit)

    @classmethod
    def from_epsilon(cls, x, Q2, epsilon, dis_limit=True):
        """Fix epsilon directly (as in the paper's figures); y is inverted from it."""
        x, Q2, eps = (np.asarray(v, float) for v in (x, Q2, epsilon))
        gamma = 0.0 * x if dis_limit else cls._gamma(x, Q2)
        # solve eps (1-y+y^2/2+g y^2) = 1-y-g y^2 with g = gamma^2/4 for y in [0,1]
        g = gamma**2 / 4.0
        a = eps / 2.0 + eps * g + g
        b = 1.0 - eps
        c = eps - 1.0
        y = np.where(a > 0, (-b + np.sqrt(b * b - 4 * a * c)) / (2 * np.where(a > 0, a, 1)), -c / b)
        return cls(x, Q2, y, eps, gamma)

    @property
    def physical(self):
        """Mask of physical electron kinematics: 0 < y and 0 <= epsilon < 1
        (i.e. 1 - y - gamma^2 y^2/4 >= 0)."""
        return (self.y > 0) & (self.epsilon >= 0) & (self.epsilon < 1)

    @property
    def flux(self):
        r"""Flux factor 2 pi alpha^2 y^2 / (Q^4 (1-eps) x)  [GeV^-4], Eq. (2.21a)."""
        return 2 * np.pi * ALPHA_EM**2 * self.y**2 / (self.Q2**2 * (1 - self.epsilon) * self.x)


# ===========================================================================
# deuteron polarization
# ===========================================================================
@dataclass
class Polarization:
    r"""Invariant deuteron polarization parameters (paper Eq. (2.24), Part I Sec. III B).

    {S_L, S_T, phi_S}, {T_LL, T_LT, T_TT, phi_TL, phi_TT}; angles in radians,
    measured relative to the lepton scattering plane like phi_p.
    Default: unpolarized.
    """
    SL: float = 0.0
    ST: float = 0.0
    phi_S: float = 0.0
    TLL: float = 0.0
    TLT: float = 0.0
    TTT: float = 0.0
    phi_TL: float = 0.0
    phi_TT: float = 0.0

    # ---------------------------------------------------------------------
    @staticmethod
    def W(Lam):
        """W(Lambda) = (1, -2, 1) for Lambda = (+1, 0, -1), paper Eq. (2.28c)."""
        return {1: 1.0, 0: -2.0, -1: 1.0}[int(Lam)]

    @classmethod
    def along_axis(cls, theta_N, phi_N=0.0, Pz=0.0, Pzz=0.0):
        r"""Mixed state with vector/tensor polarization Pz, Pzz along axis N(theta_N, phi_N).

        ``Pz = n_+ - n_-``, ``Pzz = n_+ + n_- - 2 n_0`` (occupation numbers of the
        Lambda = +1, 0, -1 states along N). Linear superposition of Part I
        Eq. (pure_state_invariant), so ``along_axis(th, ph, Pz=L, Pzz=W(L))`` is the
        pure state Lambda = L. theta_N is the polar angle of N w.r.t. e_z (the
        collinear axis, opposite to q) in the deuteron rest frame.
        """
        c, s = np.cos(theta_N), np.sin(theta_N)
        return cls(SL=Pz * c, ST=Pz * s, phi_S=phi_N,
                   TLL=Pzz / 6.0 * (3 * c * c - 1.0), TLT=Pzz / 2.0 * s * c,
                   TTT=Pzz / 2.0 * s * s, phi_TL=phi_N, phi_TT=phi_N)

    @classmethod
    def pure_state(cls, Lam, theta_N, phi_N=0.0):
        """Pure spin state Lambda in {+1, 0, -1} along N(theta_N, phi_N)."""
        return cls.along_axis(theta_N, phi_N, Pz=float(Lam), Pzz=cls.W(Lam))

    # --- polarization defined w.r.t. the lepton beam (Part I Sec. V A) ------
    @staticmethod
    def beam_axis_angles(kin: Kinematics, orientation="parallel", psi_N=0.0):
        r"""(theta_N, phi_N) of the polarization axis for beam-referenced setups.

        ``"parallel"``: N opposite to the incoming lepton momentum in the deuteron
        rest frame (Lambda = +1 means spin opposite to the beam).
        ``"perp"``: N perpendicular to the beam, at angle psi_N from the lepton
        plane (Part I Fig. 2). Uses kin.gamma (0 in the DIS limit) and kin.y.
        """
        g, y = kin.gamma, kin.y
        root = np.sqrt(np.maximum(1 - y - g * g * y * y / 4, 0.0))
        norm = np.sqrt(1 + g * g)
        if orientation == "parallel":
            cos_t = (1 + g * g * y / 2) / norm
            sin_t = g * root / norm
            return np.arctan2(sin_t, cos_t), 0.0 * cos_t
        if orientation in ("perp", "perpendicular"):
            cos_t = g * root / norm * np.cos(psi_N)
            st_cp = (1 + g * g * y / 2) / norm * np.cos(psi_N)
            st_sp = np.sin(psi_N) + 0.0 * cos_t
            sin_t = np.hypot(st_cp, st_sp)
            return np.arctan2(sin_t, cos_t), np.arctan2(st_sp, st_cp)
        raise ValueError("orientation must be 'parallel' or 'perp'")

    @classmethod
    def beam(cls, kin: Kinematics, orientation="parallel", Pz=0.0, Pzz=0.0, psi_N=0.0):
        """Mixed state (Pz, Pzz) along a beam-referenced axis; see :meth:`beam_axis_angles`."""
        th, ph = cls.beam_axis_angles(kin, orientation, psi_N)
        return cls.along_axis(th, ph, Pz=Pz, Pzz=Pzz)


# ===========================================================================
# main class
# ===========================================================================
_SF_NAMES = (
    # unpolarized deuteron, Eq. (5.12)
    "UU_T", "UU_L", "UU_cos_phi", "UU_cos_2phi",
    # T_LL, Eq. (5.14)
    "UTLL_T", "UTLL_L", "UTLL_cos_phi", "UTLL_cos_2phi",
    # T_LT, Eq. (5.15): cos(phi-phiTL) T,L ; cos(phiTL)=cos(2phi-phiTL) ; cos(phi+phiTL)=cos(3phi-phiTL)
    "UTLT_T", "UTLT_L", "UTLT_tw3", "UTLT_tw4",
    # T_TT, Eq. (5.16): cos(2phi-2phiTT) T,L ; cos(phi-2phiTT)=cos(3phi-2phiTT) ; cos2phiTT=cos(4phi-2phiTT)
    "UTTT_T", "UTTT_L", "UTTT_tw3", "UTTT_tw4",
    # polarized electron, vector-polarized deuteron, Eqs. (5.26)-(5.27)
    "LSL", "LST_cos_phi_m_phiS", "LSL_cos_phi", "LST_cos_phiS", "LST_cos_2phi_m_phiS",
)


class TaggedDIS:
    r"""Tagged DIS on the polarized deuteron in the impulse approximation.

    Parameters
    ----------
    wavefunction : str or DeuteronWavefunction
        ``"AV18"`` (default), ``"CDBonn"``, ``"Paris"``.
    structure : NucleonStructure, optional
        Nucleon structure functions; default SLAC F2 (R=0.18) + GRSV2000 g1 + WW g2.
    tagged : {"p", "n"}
        Detected spectator nucleon; the *other* nucleon is the active one whose
        structure functions are used. The LF distributions are the same (isospin
        symmetric deuteron wave function, average nucleon mass).

    All public methods are numpy-vectorized over the kinematic arguments.
    """

    def __init__(self, wavefunction="AV18", structure: NucleonStructure | None = None,
                 tagged="p"):
        self.lf = LFDistributions(wavefunction)
        self.structure = structure if structure is not None else default_structure()
        if tagged not in ("p", "n"):
            raise ValueError("tagged must be 'p' or 'n'")
        self.tagged = tagged
        self.active = "n" if tagged == "p" else "p"

    # ------------------------------------------------------------------
    def nucleon_variables(self, x, Q2, alpha_p):
        r"""x_n = x/(2-alpha_p) (Eq. (5.9)) and gamma_n = 2 x_n m / Q (Eq. (5.22))."""
        xn = np.asarray(x, float) / (2.0 - np.asarray(alpha_p, float))
        gamma_n = 2.0 * xn * M_NUCLEON / np.sqrt(np.asarray(Q2, float))
        return xn, gamma_n

    def structure_functions(self, x, Q2, alpha_p, pT):
        r"""All IA tagged deuteron structure functions [GeV^-2] at (x, Q2, alpha_p, pT).

        Returns a dict with the keys in ``_SF_NAMES`` (see source for the map to the
        paper's notation) plus the neutron densities ``"P"`` and nucleon inputs.
        Kinematically forbidden points (alpha_p >= 2 - x) give 0.

        Naming (paper notation -> key):

        ===========================================  ===================
        F_[UU,T]D, F_[UU,L]D                          UU_T, UU_L
        F_[UU]D^{cos phi_p}, ^{cos 2 phi_p}           UU_cos_phi, UU_cos_2phi
        F_[UT_LL,T/L]D, ^{cos phi_p}, ^{cos 2phi_p}   UTLL_T, UTLL_L, ...
        F_[UT_LT,T/L]D^{cos(phi_p-phi_TL)}            UTLT_T, UTLT_L
        F^{cos phi_TL} = F^{cos(2phi_p-phi_TL)}       UTLT_tw3
        F^{cos(phi_p+phi_TL)} = F^{cos(3phi_p-phi_TL)} UTLT_tw4
        F_[UT_TT,T/L]D^{cos(2phi_p-2phi_TT)}          UTTT_T, UTTT_L
        F^{cos(phi_p-2phi_TT)} = F^{cos(3phi_p-2phi_TT)} UTTT_tw3
        F^{cos 2phi_TT} = F^{cos(4phi_p-2phi_TT)}     UTTT_tw4
        F_[LS_L]D                                     LSL
        F_[LS_T]D^{cos(phi_p-phi_S)}                  LST_cos_phi_m_phiS
        F_[LS_L]D^{cos phi_p}                         LSL_cos_phi
        F_[LS_T]D^{cos phi_S}                         LST_cos_phiS
        F_[LS_T]D^{cos(2phi_p-phi_S)}                 LST_cos_2phi_m_phiS
        ===========================================  ===================

        (Eq. (5.16c) of the draft writes the partner of cos(phi_p-2phi_TT) as
        cos(3phi_p-phi_TT); the master formula, Part I Eq. (4.39), has
        cos(3phi_p-2phi_TT), which is what is used here.)
        """
        x, Q2, alpha_p, pT = np.broadcast_arrays(*(np.asarray(v, float)
                                                   for v in (x, Q2, alpha_p, pT)))
        allowed = (alpha_p > 0) & (alpha_p < 2.0 - x)
        a_safe = np.where(allowed, alpha_p, 1.0)
        P = self.lf(a_safe, pT)
        xn, gamma_n = self.nucleon_variables(x, Q2, a_safe)
        xn = np.where(allowed, xn, 0.5)
        Q = np.sqrt(Q2)

        st = self.structure
        FTn, FLn = st.FT_FL(xn, Q2, self.active)
        F2n = st.F1F2(xn, Q2, self.active)[1]      # = x_n (FT + FL) in the DIS limit, Eq. (5.10b)
        g1 = st.g1(xn, Q2, self.active)
        g1g2 = st.g1_plus_g2(xn, Q2, self.active)
        FLSLn = 2.0 * g1                           # Eq. (5.24a)
        FLSTn = -2.0 * gamma_n * g1g2              # Eq. (5.24b)

        h = np.where(allowed, 2.0 * TWO_2PI3 / (2.0 - a_safe), 0.0)   # 2[2(2pi)^3]/(2-alpha_p)
        tw3 = 2.0 * pT / Q * F2n                   # (2 x_n pT/Q)(FT+FL)
        tw4 = 2.0 * xn * pT**2 / Q2 * F2n          # (2 x_n^2 pT^2/Q^2)(FT+FL)

        sf = {}
        # unpolarized deuteron (5.12)
        sf["UU_T"] = h * P.UU * FTn
        sf["UU_L"] = h * P.UU * FLn
        sf["UU_cos_phi"] = h * P.UU * tw3
        sf["UU_cos_2phi"] = h * P.UU * tw4
        # T_LL (5.14)
        pll = SQRT32 * P.TLL_U
        sf["UTLL_T"] = h * pll * FTn
        sf["UTLL_L"] = h * pll * FLn
        sf["UTLL_cos_phi"] = h * pll * tw3
        sf["UTLL_cos_2phi"] = h * pll * tw4
        # T_LT (5.15)
        sf["UTLT_T"] = h * SQRT2 * P.TLT_U * FTn
        sf["UTLT_L"] = h * SQRT2 * P.TLT_U * FLn
        sf["UTLT_tw3"] = h / SQRT2 * P.TLT_U * tw3
        sf["UTLT_tw4"] = h / SQRT2 * P.TLT_U * tw4
        # T_TT (5.16)
        sf["UTTT_T"] = h / SQRT2 * P.TTT_U * FTn
        sf["UTTT_L"] = h / SQRT2 * P.TTT_U * FLn
        sf["UTTT_tw3"] = h / (2 * SQRT2) * P.TTT_U * tw3
        sf["UTTT_tw4"] = h / (2 * SQRT2) * P.TTT_U * tw4
        # vector polarized, polarized electron (5.26), (5.27)
        sf["LSL"] = h * P.SL_SL * FLSLn
        sf["LST_cos_phi_m_phiS"] = h * P.ST_SL * FLSLn
        sf["LSL_cos_phi"] = h * P.SL_ST * FLSTn
        sf["LST_cos_phiS"] = h * 0.5 * (P.ST_ST_perp + P.ST_ST_par) * FLSTn
        sf["LST_cos_2phi_m_phiS"] = h * 0.5 * (P.ST_ST_par - P.ST_ST_perp) * FLSTn

        sf["P"] = P
        sf["nucleon"] = dict(xn=xn, gamma_n=gamma_n, FT=FTn, FL=FLn, F2=F2n,
                             g1=g1, g1_plus_g2=g1g2, allowed=allowed)
        return sf

    # ------------------------------------------------------------------
    @staticmethod
    def assemble(sf, eps, phi_p=0.0, lambda_e=0.0, pol: Polarization | None = None,
                 phi_averaged=False):
        r"""F_U + F_S + F_T from the structure functions (Part I master formula).

        Parameters
        ----------
        sf : dict from :meth:`structure_functions`
        eps : virtual photon polarization parameter epsilon
        phi_p : spectator azimuth w.r.t. the lepton plane [rad]
        lambda_e : electron helicity in {-1/2, 0, +1/2} (0 = unpolarized beam)
        pol : :class:`Polarization` (default unpolarized deuteron)
        phi_averaged : if True return <F>_1 = int dphi_p/2pi F (Part I
            Eq. (cross_section_harmonic_1)); phi_p is then ignored.

        Only the terms that are nonzero in the IA are included (the T-odd
        structures vanish, Eqs. (5.17), (5.28)).
        """
        pol = pol or Polarization()
        e = np.clip(np.asarray(eps, float), 0.0, 1.0)   # unphysical eps is masked by the callers
        l2 = 2.0 * np.asarray(lambda_e, float)
        a1 = np.sqrt(2 * e * (1 + e))       # sqrt(2 eps (1+eps))
        b0 = np.sqrt(1 - e * e)             # sqrt(1-eps^2)
        b1 = np.sqrt(2 * e * (1 - e))       # sqrt(2 eps (1-eps))
        cos = np.cos
        if phi_averaged:
            F = (sf["UU_T"] + e * sf["UU_L"]
                 + pol.SL * l2 * b0 * sf["LSL"]
                 + pol.ST * l2 * b1 * cos(pol.phi_S) * sf["LST_cos_phiS"]
                 + pol.TLL * (sf["UTLL_T"] + e * sf["UTLL_L"])
                 + pol.TLT * a1 * cos(pol.phi_TL) * sf["UTLT_tw3"]
                 + pol.TTT * e * cos(2 * pol.phi_TT) * sf["UTTT_tw4"])
            return F

        p = np.asarray(phi_p, float)
        FU = (sf["UU_T"] + e * sf["UU_L"] + a1 * cos(p) * sf["UU_cos_phi"]
              + e * cos(2 * p) * sf["UU_cos_2phi"])
        FS = (pol.SL * l2 * (b0 * sf["LSL"] + b1 * cos(p) * sf["LSL_cos_phi"])
              + pol.ST * l2 * (b0 * cos(p - pol.phi_S) * sf["LST_cos_phi_m_phiS"]
                               + b1 * cos(pol.phi_S) * sf["LST_cos_phiS"]
                               + b1 * cos(2 * p - pol.phi_S) * sf["LST_cos_2phi_m_phiS"]))
        tl, tt = pol.phi_TL, pol.phi_TT
        FT = (pol.TLL * (sf["UTLL_T"] + e * sf["UTLL_L"] + a1 * cos(p) * sf["UTLL_cos_phi"]
                         + e * cos(2 * p) * sf["UTLL_cos_2phi"])
              + pol.TLT * (cos(p - tl) * (sf["UTLT_T"] + e * sf["UTLT_L"])
                           + a1 * (cos(tl) + cos(2 * p - tl)) * sf["UTLT_tw3"]
                           + e * (cos(p + tl) + cos(3 * p - tl)) * sf["UTLT_tw4"])
              + pol.TTT * (cos(2 * p - 2 * tt) * (sf["UTTT_T"] + e * sf["UTTT_L"])
                           + a1 * (cos(p - 2 * tt) + cos(3 * p - 2 * tt)) * sf["UTTT_tw3"]
                           + e * (cos(2 * tt) + cos(4 * p - 2 * tt)) * sf["UTTT_tw4"]))
        return FU + FS + FT

    # ------------------------------------------------------------------
    def reduced_cross_section(self, kin: Kinematics, alpha_p, pT, phi_p=0.0,
                              lambda_e=0.0, pol: Polarization | None = None,
                              phi_averaged=False):
        r"""sigma_red,D = (x/2)(F_U + F_S + F_T)  [GeV^-2], paper Eq. (2.21b).

        Differential in dGamma_p, i.e. ``int dGamma_p sigma_red ~ F2n - (1-eps)FLn``
        (Eq. (2.22)).
        """
        sf = self.structure_functions(kin.x, kin.Q2, alpha_p, pT)
        F = self.assemble(sf, kin.epsilon, phi_p, lambda_e, pol, phi_averaged)
        return np.where(kin.physical, 0.5 * kin.x * F, 0.0)

    def dsigma(self, kin: Kinematics, alpha_p, pT, phi_p=0.0, lambda_e=0.0,
               pol: Polarization | None = None, phi_averaged=False, nb=True):
        r"""d sigma / (dx dQ^2 (d alpha_p/alpha_p) d^2 p_pT)  [nb/GeV^4], psi_l' integrated.

        From Eq. (2.21a): flux * sigma_red / [2(2pi)^3]. With ``phi_averaged`` the
        cross section is averaged over phi_p (this is the quantity of Fig. 9).
        Set ``nb=False`` for natural units GeV^-6.
        """
        sred = self.reduced_cross_section(kin, alpha_p, pT, phi_p, lambda_e, pol, phi_averaged)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(kin.physical, kin.flux * sred / TWO_2PI3, 0.0)
        return out * GEV2_TO_NB if nb else out

    # ------------------------------------------------------------------
    # spin asymmetries (Part I Sec. V D) built from pure states
    # ------------------------------------------------------------------
    def _state_F(self, sf, kin, theta_N, phi_N, Lam, lambda_e, phi_p, phi_averaged):
        pol = Polarization.pure_state(Lam, theta_N, phi_N)
        return self.assemble(sf, kin.epsilon, phi_p, lambda_e, pol, phi_averaged)

    def vector_asymmetry(self, kin: Kinematics, alpha_p, pT, orientation="parallel",
                         psi_N=0.0, phi_p=0.0, phi_averaged=True):
        r"""Two-state double-spin asymmetry A^V (Part I Eq. (double_spin_asymmetry)).

        [s(++) - s(-+) - s(+-) + s(--)] / [sum], s(lambda_e, Lambda) for
        lambda_e = +-1/2 and deuteron Lambda = +-1 along the beam-referenced axis.
        In the DIS limit and IA, ``orientation="parallel"`` gives Eq. (6.4).
        """
        sf = self.structure_functions(kin.x, kin.Q2, alpha_p, pT)
        th, ph = Polarization.beam_axis_angles(kin, orientation, psi_N)
        s = {(le, L): self._state_F(sf, kin, th, ph, L, le, phi_p, phi_averaged)
             for le in (0.5, -0.5) for L in (1, -1)}
        num = s[(0.5, 1)] - s[(-0.5, 1)] - s[(0.5, -1)] + s[(-0.5, -1)]
        den = s[(0.5, 1)] + s[(-0.5, 1)] + s[(0.5, -1)] + s[(-0.5, -1)]
        return num / den

    def tensor_asymmetry(self, kin: Kinematics, alpha_p, pT, orientation="parallel",
                         psi_N=0.0, phi_p=0.0, phi_averaged=True):
        r"""Tensor asymmetry A^T = [s(+) + s(-) - 2 s(0)] / [s(+) + s(-) + s(0)], Part I Eq. (AT_gen).

        Beam-helicity averaged. In the DIS limit and IA, parallel gives Eq. (6.12),
        perpendicular Eq. (6.16); with ``phi_averaged=False`` the phi_p-dependent
        asymmetry of Fig. 14.
        """
        sf = self.structure_functions(kin.x, kin.Q2, alpha_p, pT)
        th, ph = Polarization.beam_axis_angles(kin, orientation, psi_N)
        s = {L: self._state_F(sf, kin, th, ph, L, 0.0, phi_p, phi_averaged) for L in (1, 0, -1)}
        return (s[1] + s[-1] - 2 * s[0]) / (s[1] + s[-1] + s[0])
