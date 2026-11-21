r"""Unpolarized inclusive structure functions and (virtual-)photon absorption cross sections.

One layer, in the variables (W, Q^2), for the nucleon and the deuteron, valid from real
photons (Q^2 = 0) through the resonance region to DIS. The same model object serves
real photoproduction and, through :class:`AsNucleonStructure`, the tagged-DIS code.

The F_T, F_L and deuteron conventions are those of W. Cosyn and C. Weiss,
Phys. Rev. C 114, 025201 (2026), arXiv:2603.23699 and Phys. Rev. C 114, 025202 (2026),
arXiv:2603.23700.

Main entry point::

    from taggedpd.inclusive import inclusive_sf
    sigma = inclusive_sf(W, "n")                        # sigma_T(gamma n -> X) [microbarn] at Q^2 = 0
    F1 = inclusive_sf(W, "p", quantity="F1", Q2=0.1)
    FL = inclusive_sf(W, "p", quantity="F_L", Q2=2.0)

``photoproduction`` is an alias of :func:`inclusive_sf`.

Architecture
------------
* :class:`InclusiveModel` -- interface: ``domain``, ``F1`` or ``sigma_T``, optionally
  ``F1F2``; everything else (sigma_L, R, F_T, F_L, ...) is derived in one place,
  :meth:`InclusiveModel.evaluate`. (``PhotoabsorptionModel`` is an alias.)
* Concrete models: :class:`ChristyBosted` (2009 fits), :class:`ChristyBosted2021`
  (PRELIMINARY), :class:`BelowThreshold`, and :class:`FromNucleonStructure`, which wraps
  any (x, Q^2) provider of :mod:`taggedpd.nucleon` (SLAC/Bodek, LHAPDF, ...) as a
  (W, Q^2) model with a kinematic domain.
* Combinators: :class:`Piecewise` (first model whose domain contains the point) and
  :class:`Blend` (linear transition in W).
* :class:`AsNucleonStructure` exposes any model (e.g. a :class:`Piecewise` with
  Christy-Bosted in the resonance region and SLAC/LHAPDF in DIS) as an (x, Q^2)
  :class:`~taggedpd.nucleon.NucleonStructure` for :class:`~taggedpd.crosssection.TaggedDIS`;
  combine it with a polarized provider via :class:`~taggedpd.nucleon.CombinedStructure`.
  :func:`resonance_dis_model` builds that combination.
  :class:`ChristyBostedStructure` is ``AsNucleonStructure`` of a single Christy-Bosted fit.

The default model for :func:`inclusive_sf` is :func:`default_photoabsorption`, a
:class:`Piecewise` combination:

1. :class:`ChristyBosted` inside the published fit ranges;
2. the same Christy-Bosted functions *extrapolated* between the pion-production
   threshold and the lower edge of the fit range (a placeholder for a dedicated
   low-W parametrization, see below);
3. :class:`BelowThreshold`: sigma_T = 0 for the free nucleon below
   W = m_N + m_pi (hadronic photoabsorption needs a pion; Compton is neglected).

Outside every piece the result is NaN (``outside="nan"``; also "zero" or
"raise"), so that missing kinematic coverage is never silent. Pass
``return_source=True`` to get, per point, the name of the piece that was used.

``default_photoabsorption(fit="2021")`` uses instead the **PRELIMINARY** 2021 fit of
M. E. Christy (:class:`ChristyBosted2021`, port of SIMC's ``F1F2IN21_v1.0.f``): it has
its own neutron fit and covers W^2 < 20 GeV^2, Q^2 < 30 GeV^2, but it is unpublished.
Constructing it emits a :class:`~taggedpd.christy_bosted21.PreliminaryFitWarning`.
The 2009 fits stay the default.

Adding a parametrization
------------------------
Subclass :class:`InclusiveModel`, implement ``sigma_T`` *or* ``F1`` (the
other follows from Hand's convention), ``domain``, and ``F1F2`` if F2 is known
(needed for F_L-type quantities at Q^2 > 0), then put it in a
:class:`Piecewise` list. Earlier entries win where domains overlap, so a
dedicated low-W model or a Regge-type high-W fit is simply placed *before* the
Christy-Bosted entries::

    class MyRegge(InclusiveModel):
        name = "Regge"
        targets = ("p", "n")
        def domain(self, W, Q2, target):
            return (W >= 2.5) & (Q2 == 0)
        def sigma_T(self, W, Q2, target):
            s = W**2
            return X * s**eps + Y * s**(-eta)           # microbarn

    model = Piecewise([MyRegge(), *default_photoabsorption().models])
    inclusive_sf(W, "p", model=model)

A smooth transition between two pieces can be had by a model whose ``domain``
is the overlap region and whose ``sigma_T`` interpolates between the two (see
:class:`Blend`).

Conventions
-----------
GeV, cross sections in **microbarn** (alpha = 1/137.036, 2021 fit 1/137.03599;
1 GeV^-2 = 389.4 microbarn). The deuteron result is the *inelastic* (pion-production)
part only: photodisintegration gamma d -> p n is not included.

Targets and kinematic variables. All variables are Lorentz invariants, described in the
target rest frame but computable in any frame (fixed target or collider). Target: a free nucleon for
"p", "n"; the deuteron for "d", whose structure functions are the deuteron's own (arXiv:2603.23700),
not divided by 2 ("per nucleus"). Bound nucleons move (Fermi motion) and each has its
own p_N.q and gamma*N mass; the deuteron fits integrate over that internally. None of
the variables below refers to a bound nucleon.

=========== ================================================================ =========
symbol      meaning                                                          range
=========== ================================================================ =========
M           the fit's nucleon mass (:meth:`InclusiveModel.kin_mass`): 2009
            0.93828 (flux 0.9382727); 2021 0.938272 (p, n), average (d)
M_T         target mass: M for p, n; M_D (:data:`~taggedpd.constants.M_DEUTERON`)
            for d (:meth:`InclusiveModel.sf_mass`)
nu          photon energy in the target rest frame, the same quantity for
            every target: p_T.q/M_T is its invariant form for a nucleon or
            deuteron target, valid in any frame. In a collider it is NOT the
            photon energy in the collider frame; compute p_D.q/M_D from the
            four-vectors. nu = (W^2 - M^2 + Q^2)/2M, so at given (W, Q^2)
            all targets have the same nu (and the same K below)
W           input, always converted to nu by W^2 = M^2 + 2 M nu - Q^2. p, n:
            the gamma*N invariant mass. d: not the mass of any physical
            system, just another way of specifying nu: W^2 = (p_N + q)^2 for
            a free nucleon moving with the deuteron, p_N = (M/M_D) p_D (at
            rest in the deuteron rest frame), as in the fits. NOT the
            gamma*d mass W_D^2 = M_D^2 + 2 M_D nu - Q^2, nor that of a bound
            nucleon.
            :func:`W_from_nu` / :func:`nu_from_W` convert.
x_N         Q^2/(2 M nu), free-nucleon Bjorken variable, used for p, n       (0, 1]
x_D         Q^2/(2 M_D nu) = Q^2/(2 p_D.q), deuteron Bjorken variable,       (0, 1]
            used for d
x (tagged)  2 x_D, tagged-DIS variable of arXiv:2603.23700 (TaggedDIS);      (0, 2]
            NOT used here
gamma^2     Q^2/nu^2 = 4 M^2 x_N^2/Q^2 = 4 M_D^2 x_D^2/Q^2
K           Hand's equivalent photon energy for a free nucleon,
            (W^2 - M^2)/2M = nu - Q^2/2M, used for all targets, d included
=========== ================================================================ =========

W1, W2 are the hadronic tensor of the target (free nucleon for p, n; deuteron for d).
Below, x_T = x_N for p, n and x_T = x_D for d.

sigma_T, sigma_L are virtual-photon absorption cross sections in Hand's convention,
*not* measurable photoproduction cross sections. They are defined by writing the
electron cross section as a flux Gamma times sigma_T + eps sigma_L. In invariants
(l, p_T lepton and target momenta, lepton mass neglected)::

    d^2 sigma/(dx_T dQ^2) = Gamma (sigma_T + eps sigma_L)
                          = 2 pi alpha^2 y^2/(Q^4 (1 - eps)) (F_T + eps F_L),
    Gamma = alpha M_T K y^2/(pi Q^4 (1 - eps)),
    sigma_{T,L} = 2 pi^2 alpha F_{T,L}/(M_T K),
    y = p_T.q/p_T.l,  eps = (1 - y - gamma^2 y^2/4)/(1 - y + y^2/2 + gamma^2 y^2/4),
    K = (W^2 - M^2)/2M = nu - Q^2/2M  (W^2 = (p+q)^2 for a free nucleon; for d, W just specifies nu).

K cancels in Gamma sigma: only F_T, F_L (W1, W2) are physical. For d this is arXiv:2603.23700's
d sigma (dx/2 = dx_D). For real photons (Q^2 -> 0): K/nu -> 1, gamma^2 -> 0,
sigma_L -> 0, and sigma_T is the measured total photoabsorption cross section. At
Q^2 > 0 the normalization of sigma_T depends on the choice of K (Gamma sigma_T does not).

Structure-function conventions (``quantity=``)
----------------------------------------------
=============== ============================================ ==================================
quantity        definition                                    remarks
=============== ============================================ ==================================
"sigma_T"       4 pi^2 alpha W1/K [microbarn]                  Hand, free-nucleon K (above)
"sigma_L"       R sigma_T [microbarn]
"R"             sigma_L/sigma_T = F_L/F_T
"F1"            M_T W1 = (M_T/M) sigma_T (W^2 - M^2)/(8 pi^2 alpha)  p, n: the fits' native F1
"F2"            nu W2                                          0 at Q^2 = 0
"F_T"           F_[UU,T] = 2 F1                                arXiv:2603.23699/23700 convention
"F_L"           F_[UU,L] = (1 + gamma^2) F2/x_T - 2 F1         exact (target-mass) relation
                = R F_T
"F_L_scaling"   F2/x_T - 2 F1                                  scaling limit (gamma -> 0),
                = F_T (R - gamma^2)/(1 + gamma^2)              arXiv:2603.23700 Eq. (5.10)
"xF_L"          (1 + gamma^2) F2 - 2 x_T F1 = x_T F_[UU,L]     the "F_L" of Christy-Bosted /
                                                               SIMC and of many DIS papers
=============== ============================================ ==================================

Notes:

* Deuteron (arXiv:2603.23700): F1^D = M_D W1, F2^D = nu W2, Callan-Gross F2^D = 2 x_D F1^D.
  For free nucleons at rest in the deuteron (no Fermi motion, no binding) the hadronic
  tensors add at the same (nu, Q^2), i.e. at the same input W:
  W1^D ~ W1p + W1n and W2^D ~ W2p + W2n. In terms of the F's, with x_N = (M_D/M) x_D ~ 2 x_D:
  F2^D(x_D) ~ F2p(x_N) + F2n(x_N) and F1^D(x_D) ~ (M_D/M) [F1p(x_N) + F1n(x_N)]. So the
  F1-type deuteron functions are about **4x** the nucleon average, F2 about 2x;
  sigma_T, sigma_L add like W1, W2. The Christy-Bosted deuteron fit adds Fermi smearing and
  MEC: W1^D/(W1p + W1n) ~ 0.97-0.99 at W = 2-2.8, but 0.65-0.85 at the Delta.
* The low-level ports (:mod:`taggedpd.christy_bosted`, :mod:`taggedpd.christy_bosted21`)
  keep the Fortran convention for d: per nucleus, but F1 = M W1 with x_N in (0, 2], i.e.
  F1^D = (M_D/M) F1_Fortran. F2, R, gamma, x F1 and "xF_L" agree between the two.
* Deuteron flux: Hand's K could be defined with the deuteron as target,
  K_D = (W_D^2 - M_D^2)/2M_D = nu - Q^2/2M_D. The fits (and this module) use the
  free-nucleon K for d too. Then sigma_T(K_D) = sigma_T K/K_D: equal at Q^2 = 0, ~0.73 at
  Q^2 = 1, W = 1.5. W1, W2, R and all F's are independent of this choice.
* F_T and F_L have the dimension of F1. Elsewhere one also finds F_T = 2xF1 with
  F_L = (1+gamma^2)F2 - 2xF1 (= "xF_L" here), such that (1+gamma^2) F2 = F_T + F_L:
  x times the convention used here.
* "F_L_scaling" equals F_L only when gamma^2 << 1. In the resonance region at low Q^2,
  gamma^2 is *not* small (e.g. ~0.5 at Q^2 = 1, W = 1.5), and F_L_scaling can even be
  negative where gamma^2 > R. Use "F_L" for the physical longitudinal structure function.
* At Q^2 = 0 all F_L-type quantities, F2, R and sigma_L are returned as exactly 0
  (their limits; x = 0 there). Models that only provide sigma_T/F1 (e.g. a
  user-supplied Regge form) give F2- and F_L-type quantities only at Q^2 = 0 and
  raise ``NotImplementedError`` otherwise.
* Model classes: ``F1``/``F1F2`` of an :class:`InclusiveModel` return the convention
  above (for d: M_D W1). A user model for d that implements ``sigma_T`` gets this
  automatically; one that implements ``F1`` must return M_D W1.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import christy_bosted as cb
from . import christy_bosted21 as cb21

from .constants import M_DEUTERON, M_NEUTRON, M_PROTON
from .nucleon import NucleonStructure

__all__ = ["InclusiveModel", "PhotoabsorptionModel", "ChristyBosted", "ChristyBosted2021",
           "BelowThreshold", "FromNucleonStructure", "AsNucleonStructure", "ChristyBostedStructure",
           "Piecewise", "Blend", "default_photoabsorption", "resonance_dis_model",
           "inclusive_sf", "photoproduction", "W_from_nu", "nu_from_W", "PION_THRESHOLD",
           "QUANTITIES"]

_M = cb.MP_CB
#: free-nucleon pion-production threshold W = m_p + m_pi0 [GeV], as in the fits
PION_THRESHOLD = cb.MP_CB + 0.135


def W_from_nu(nu, Q2=0.0, mass=_M):
    """W for photon energy ``nu`` in the target rest frame (invariant p_T.q/M_T): W^2 = M^2 + 2 M nu - Q^2."""
    return np.sqrt(np.maximum(mass**2 + 2.0 * mass * np.asarray(nu, float) - Q2, 0.0))


def nu_from_W(W, Q2=0.0, mass=_M):
    """Inverse of :func:`W_from_nu`."""
    return (np.asarray(W, float) ** 2 - mass**2 + Q2) / (2.0 * mass)


def _norm_target(target):
    t = {"p": "p", "proton": "p", "n": "n", "neutron": "n", "d": "d", "deuteron": "d"}.get(target)
    if t is None:
        raise ValueError(f"target must be 'p', 'n' or 'd', got {target!r}")
    return t


# ===========================================================================
# interface
# ===========================================================================
def _sf_factor(target, M):
    """F1 = M_T W1 relative to M W1: 1 for p, n; M_D/M for d (arXiv:2603.23700 convention)."""
    return M_DEUTERON / M if _norm_target(target) == "d" else 1.0


class InclusiveModel:
    """Interface for a parametrization of sigma_T (equivalently F1) at given (W, Q^2).

    Subclasses set ``name`` and ``targets`` and implement :meth:`domain` and at
    least one of :meth:`sigma_T`, :meth:`F1`. All methods take broadcastable
    arrays ``W``, ``Q2`` [GeV, GeV^2] and a target string.
    """

    name = "abstract"
    #: targets the model can describe, subset of ("p", "n", "d")
    targets: tuple = ()
    #: mass in Hand's flux factor, F1 <-> sigma_T
    flux_mass = _M

    def domain(self, W, Q2, target):
        """Boolean mask: where this model should be used."""
        raise NotImplementedError

    def sigma_T(self, W, Q2, target):
        """Transverse cross section [microbarn] (per nucleus for d)."""
        if type(self).F1 is InclusiveModel.F1:
            raise NotImplementedError(f"{type(self).__name__} implements neither sigma_T nor F1")
        W = np.asarray(W, float)
        F1 = self.F1(W, Q2, target) / _sf_factor(target, self.flux_mass)
        return cb.sigmaT_from_F1(F1, W * W, self.flux_mass)

    def F1(self, W, Q2, target):
        """F1 = M_T W1 of the target: free nucleon for p, n (M_T = M); deuteron for d (M_T = M_D, arXiv:2603.23700)."""
        if type(self).sigma_T is InclusiveModel.sigma_T:
            raise NotImplementedError(f"{type(self).__name__} implements neither sigma_T nor F1")
        W = np.asarray(W, float)
        F1 = cb.F1_from_sigmaT(self.sigma_T(W, Q2, target), W * W, self.flux_mass)
        return F1 * _sf_factor(target, self.flux_mass)

    def kin_mass(self, target):
        """Nucleon mass M in the conversion W -> nu = (W^2 - M^2 + Q^2)/2M
        (also for d, where W only specifies nu), and in gamma^2 = Q^2/nu^2."""
        return _M

    def sf_mass(self, target):
        """Target mass M_T in F1 = M_T W1 and x = Q^2/(2 M_T nu): kin_mass for p, n; M_D for d."""
        return M_DEUTERON if _norm_target(target) == "d" else self.kin_mass(target)

    def F1F2(self, W, Q2, target):
        """(F1, F2). Default for models that only provide sigma_T/F1: F2 is known only at
        Q^2 = 0 (where it vanishes); at Q^2 > 0 this raises NotImplementedError."""
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        if np.any(Q2 > 0):
            raise NotImplementedError(f"{self.name!r} provides only sigma_T/F1; F2 and F_L-type "
                                      "quantities are available only at Q2 = 0")
        return self.F1(W, Q2, target), np.zeros(W.shape)

    def evaluate(self, quantity, W, Q2, target):
        """Any quantity of :data:`QUANTITIES`, from this model's F1, F2 and :meth:`kin_mass`."""
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {sorted(QUANTITIES)}, got {quantity!r}")
        t = _norm_target(target)
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        if quantity == "sigma_T":
            return self.sigma_T(W, Q2, t)
        if quantity == "F1":
            return self.F1(W, Q2, t)
        if quantity == "F_T":
            return 2.0 * self.F1(W, Q2, t)
        F1, F2 = self.F1F2(W, Q2, t)
        if quantity == "F2":
            return F2
        M = self.kin_mass(t)
        pos = Q2 > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = (W * W - M * M + Q2) / (2.0 * M)
            x = Q2 / (2.0 * self.sf_mass(t) * nu)        # x_T: x_N for p, n; x_D for d
            g2 = Q2 / nu**2
            FL = np.where(pos, (1.0 + g2) * F2 / x - 2.0 * F1, 0.0)
            if quantity == "F_L":
                return FL
            if quantity == "F_L_scaling":
                return np.where(pos, F2 / x - 2.0 * F1, 0.0)
            if quantity == "xF_L":
                return np.where(pos, (1.0 + g2) * F2 - 2.0 * x * F1, 0.0)
            R = np.where(F1 != 0, FL / (2.0 * F1), 0.0)
        if quantity == "R":
            return R
        return R * self.sigma_T(W, Q2, t)          # sigma_L

    def supports(self, target):
        return _norm_target(target) in self.targets

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"


#: quantities :func:`inclusive_sf` / :meth:`InclusiveModel.evaluate` can return.
#: See the module docstring, "Structure-function conventions".
QUANTITIES = {
    "sigma_T": "virtual-photon transverse cross section [microbarn] = 4 pi^2 alpha W1/K (Hand, free-nucleon K = nu - Q^2/2M); real-photon cross section at Q^2 = 0",
    "sigma_L": "longitudinal cross section [microbarn] = R sigma_T",
    "R": "sigma_L / sigma_T = F_L / F_T",
    "F1": "F1 = M_T W1 = (M_T/M) sigma_T (W^2 - M^2) / (8 pi^2 alpha)  (M_T = M_D for d: per nucleus, x_D)",
    "F2": "F2 = nu W2",
    "F_T": "F_[UU,T] = 2 F1  (arXiv:2603.23699 and arXiv:2603.23700)",
    "F_L": "F_[UU,L] = (1 + gamma^2) F2 / x_T - 2 F1 = R F_T  (exact; x_T = x_N for p, n, x_D for d; gamma^2 = Q^2/nu^2)",
    "F_L_scaling": "F2 / x_T - 2 F1  (scaling-limit relation gamma -> 0, arXiv:2603.23700 Eq. (5.10))",
    "xF_L": "x_T F_[UU,L] = (1 + gamma^2) F2 - 2 x_T F1 = 2 x_T F1 R  (the 'F_L' of Christy-Bosted / SIMC, "
            "of many DIS papers)",
}


# ===========================================================================
# Christy-Bosted
# ===========================================================================
class ChristyBosted(InclusiveModel):
    r"""Christy-Bosted fits (see :mod:`taggedpd.christy_bosted` for references and details).

    p : proton fit [PRC 81, 055213 (2010)].
    n : 2 (p+n)/2 - p, with (p+n)/2 from the deuteron fit [PRC 77, 065206 (2008)].
    d : Fermi-smeared (p+n)/2 plus the fit's MEC term, per nucleus (inelastic only).

    Parameters
    ----------
    W_range : dict, optional
        target -> (W_min, W_max) where this model claims validity. Default: the
        published fit ranges, p: (1.1, 3.1); n: (1.1, 3.0) (needs both fits);
        d: (1.2, 3.0) in free-proton W.
    Q2_max : dict, optional
        target -> maximal Q^2. Default p: 8, n: 8, d: 10 GeV^2.
    binned : bool
        False (default): evaluate the resonance shapes of RESMODD at the exact W
        (smooth). True: 1 MeV W bins as in the FORTRAN (and the nucqcd C++),
        which produces a staircase in n and d: steps ~1% above W = 1.2 GeV and up to
        tens of percent within ~25 MeV of the pion threshold.
    name : str
    """

    targets = ("p", "n", "d")
    DEFAULT_W_RANGE = {"p": (1.1, 3.1), "n": (1.1, 3.0), "d": (1.2, 3.0)}
    DEFAULT_Q2_MAX = {"p": 8.0, "n": 8.0, "d": 10.0}

    def __init__(self, W_range=None, Q2_max=None, binned=False, name="Christy-Bosted"):
        self.W_range = dict(self.DEFAULT_W_RANGE, **(W_range or {}))
        self.Q2_max = dict(self.DEFAULT_Q2_MAX, **(Q2_max or {}))
        self.binned = binned
        self.name = name

    def domain(self, W, Q2, target):
        t = _norm_target(target)
        lo, hi = self.W_range[t]
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= lo) & (W <= hi) & (Q2 >= 0) & (Q2 <= self.Q2_max[t])

    def F1(self, W, Q2, target):
        return self.F1F2R(W, Q2, target)[0]

    def F1F2R(self, W, Q2, target):
        """(F1, F2, R) of F1F2IN09; for d, F1 is converted to the arXiv:2603.23700 convention M_D W1."""
        W = np.asarray(W, float)
        t = _norm_target(target)
        F1, F2, R = cb.f1f2in09(t, Q2, W * W, binned=self.binned)
        return F1 * _sf_factor(t, cb._PM), F2, R

    def sigma_T(self, W, Q2, target):
        """sigma_T [microbarn] from the fit's native F1 (Hand flux with the fit's proton mass)."""
        W = np.asarray(W, float)
        F1 = cb.f1f2in09(_norm_target(target), Q2, W * W, binned=self.binned)[0]
        return cb.sigmaT_from_F1(F1, W * W, self.flux_mass)

    def F1F2(self, W, Q2, target):
        return self.F1F2R(W, Q2, target)[:2]

    def kin_mass(self, target):
        """F1F2IN09 builds F2 with nu from PM = 0.93828 GeV, for p, n and d."""
        return cb._PM


class ChristyBosted2021(InclusiveModel):
    r"""**PRELIMINARY** Christy 2021 fit (F1F221 v1.0 prerelease 1), see :mod:`taggedpd.christy_bosted21`.

    Source: ``F1F2IN21_v1.0.f`` by M. E. Christy, distributed with JLab SIMC
    (https://github.com/JeffersonLab/simc_gfortran); deuteron/neutron fit by Christy,
    Kalantarians, Ethier, Melnitchouk, "to be published". **Unpublished**: check its
    status with the author before using it in published work; the 2009 fits
    (:class:`ChristyBosted`) are the citable alternative. Construction emits a
    :class:`~taggedpd.christy_bosted21.PreliminaryFitWarning`.

    p, n : own fits for sigma_T (and sigma_L).
    d    : p + n smeared with CD-Bonn light-cone distributions, per nucleus, inelastic only.

    Parameters
    ----------
    W2_max, Q2_max : float
        Domain (default W^2 <= 20, Q^2 <= 30 GeV^2: the range quoted in the routines'
        comments; the file header quotes W^2, Q^2 < 32).
    W_min : dict, optional
        target -> lower W limit (default: pion threshold for p, n; none for d).
    """

    targets = ("p", "n", "d")
    preliminary = True

    def __init__(self, W2_max=20.0, Q2_max=30.0, W_min=None, name="Christy 2021 (PRELIMINARY)",
                 warn=True):
        if warn:
            warnings.warn("ChristyBosted2021 is the PRELIMINARY, unpublished F1F221 fit "
                          "(SIMC F1F2IN21_v1.0.f, M. E. Christy). Check its status with the "
                          "author before using it in published work.",
                          cb21.PreliminaryFitWarning, stacklevel=2)
        self.W2_max, self.Q2_max = W2_max, Q2_max
        self.W_min = dict({"p": PION_THRESHOLD, "n": PION_THRESHOLD, "d": 0.0}, **(W_min or {}))
        self.name = name

    def domain(self, W, Q2, target):
        t = _norm_target(target)
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= self.W_min[t]) & (W * W <= self.W2_max) & (Q2 >= 0) & (Q2 <= self.Q2_max)

    def F1(self, W, Q2, target):
        return self.F1F2(W, Q2, target)[0]

    def F1F2(self, W, Q2, target):
        """(F1, F2) of F1F2IN21; for d, F1 is converted to the arXiv:2603.23700 convention M_D W1."""
        W = np.asarray(W, float)
        t = _norm_target(target)
        F1, F2 = cb21.f1f2in21(t, Q2, W * W)
        return F1 * _sf_factor(t, self.kin_mass(t)), F2

    def kin_mass(self, target):
        """SF uses M = 0.938272 for p and n; SMEARSUB the average nucleon mass for d."""
        return cb21._MAVG if _norm_target(target) == "d" else cb21._MP_SF

    def sigma_T(self, W, Q2, target):
        """sigma_T [microbarn]: the fits' own sigma_T for p, n; RESCSD's conversion for d."""
        t = _norm_target(target)
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        w2 = W * W
        if t == "p":
            return cb21.rescsp(w2, Q2)[0]
        if t == "n":
            return cb21.rescsn(w2, Q2)[0]
        F1 = cb21.rescsd(w2, Q2)[0]
        m2 = cb21._MAVG**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.3894e3 * F1 * (3.14158 * 3.14159) * (1.0 / 137.036) * 8.0 / np.abs(w2 - m2)


class BelowThreshold(InclusiveModel):
    """sigma_T = 0 for a free nucleon below the pion-production threshold (Compton neglected)."""

    name = "below pion threshold"
    targets = ("p", "n")

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W < PION_THRESHOLD) & (Q2 >= 0)

    def sigma_T(self, W, Q2, target):
        return np.zeros(np.broadcast(np.asarray(W), np.asarray(Q2)).shape)

    def F1F2(self, W, Q2, target):
        z = np.zeros(np.broadcast(np.asarray(W), np.asarray(Q2)).shape)
        return z, z.copy()


# ===========================================================================
# adapters between (x, Q^2) NucleonStructure providers and (W, Q^2) models
# ===========================================================================
def _nucleon_mass(target):
    return M_PROTON if _norm_target(target) == "p" else M_NEUTRON


class FromNucleonStructure(InclusiveModel):
    """Any (x, Q^2) :class:`~taggedpd.nucleon.NucleonStructure` (SLAC/Bodek, LHAPDF, ...) as a (W, Q^2) model.

    x = Q^2/(W^2 - M^2 + Q^2) with M the physical proton/neutron mass (the convention
    of :mod:`taggedpd.nucleon`, which also uses it in gamma^2), so that the derived
    F_T, F_L reproduce ``provider.FT_FL`` exactly for providers with ``target_mass=True``.
    Nucleons only (p, n). Real photons are not described by DIS-type fits (F2 -> 0 as
    x -> 0 at Q^2 -> 0), hence the Q^2_min default.

    Parameters
    ----------
    provider : NucleonStructure
    W_min, W_max : float
        Domain in W [GeV] (default: 0 .. infinity).
    Q2_min, Q2_max : float
        Domain in Q^2 [GeV^2] (default: 1.0 .. infinity; the point Q^2 = Q2_min is included).
    name : str, optional
    """

    targets = ("p", "n")

    def __init__(self, provider, W_min=0.0, W_max=np.inf, Q2_min=1.0, Q2_max=np.inf, name=None):
        self.provider = provider
        self.W_min, self.W_max, self.Q2_min, self.Q2_max = W_min, W_max, Q2_min, Q2_max
        self.name = name or getattr(provider, "name", type(provider).__name__)

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= self.W_min) & (W <= self.W_max) & (Q2 >= self.Q2_min) & (Q2 <= self.Q2_max)

    def kin_mass(self, target):
        return _nucleon_mass(target)

    def _x(self, W, Q2, target):
        M = _nucleon_mass(target)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(Q2 > 0, Q2 / (W * W - M * M + Q2), 0.0)

    def F1F2(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        F1, F2 = self.provider.F1F2(self._x(W, Q2, target), Q2, _norm_target(target))
        return np.broadcast_to(F1, W.shape).astype(float), np.broadcast_to(F2, W.shape).astype(float)

    def F1(self, W, Q2, target):
        return self.F1F2(W, Q2, target)[0]


class AsNucleonStructure(NucleonStructure):
    """Expose a (W, Q^2) :class:`InclusiveModel` as an (x, Q^2) :class:`~taggedpd.nucleon.NucleonStructure`.

    This is how :class:`~taggedpd.crosssection.TaggedDIS` uses Christy-Bosted (or any
    :class:`Piecewise` combination, e.g. :func:`resonance_dis_model`). Unpolarized only:
    combine with a polarized provider through :class:`~taggedpd.nucleon.CombinedStructure`.

    W^2 = M^2 + Q^2 (1/x - 1). Points with x <= 0 or x >= 1 give 0, as for the other
    providers. A :class:`Piecewise` model applies its own domains and ``outside``
    policy (NaN by default, which then propagates into cross sections: use
    ``outside="zero"`` or make sure the pieces cover the kinematics). A single model is
    evaluated without a domain check (extrapolated), like the FORTRAN-based providers.

    Parameters
    ----------
    model : InclusiveModel
    target_mass : bool
        ``FT_FL`` returns F_[UU,L] = "F_L" = (1+gamma^2)F2/x - 2F1 = R F_T (True, default)
        or the scaling-limit relation "F_L_scaling" = F2/x - 2F1 (False); cf. the
        ``target_mass`` convention of :meth:`NucleonStructure.FT_FL`.
    mass : float or dict, optional
        Nucleon mass in the x -> W conversion; default physical m_p / m_n.
    name : str, optional
    """

    def __init__(self, model, target_mass=True, mass=None, name=None):
        self.model = model
        self.target_mass = target_mass
        self._mass = mass
        self.name = name or model.name

    def mass(self, nucleon):
        if self._mass is None:
            return _nucleon_mass(nucleon)
        if isinstance(self._mass, dict):
            return self._mass[_norm_target(nucleon)]
        return self._mass

    def W(self, x, Q2, nucleon="n"):
        M = self.mass(nucleon)
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(np.maximum(M * M + np.asarray(Q2, float) * (1.0 / x - 1.0), 0.0))

    def evaluate(self, quantity, x, Q2, nucleon="n"):
        """Any :data:`QUANTITIES` entry at (x, Q^2); 0 outside 0 < x < 1."""
        x, Q2 = np.broadcast_arrays(np.asarray(x, float), np.asarray(Q2, float))
        ok = (x > 0) & (x < 1)
        out = np.zeros(x.shape)
        if np.any(ok):
            W = self.W(x[ok], Q2[ok], nucleon)
            t = _norm_target(nucleon)
            if isinstance(self.model, Piecewise):
                out[ok] = self.model._evaluate(quantity, W, Q2[ok], t)[0]
            else:
                out[ok] = self.model.evaluate(quantity, W, Q2[ok], t)
        return out

    def F1F2(self, x, Q2, nucleon="n"):
        return self.evaluate("F1", x, Q2, nucleon), self.evaluate("F2", x, Q2, nucleon)

    def FT_FL(self, x, Q2, nucleon="n"):
        return (self.evaluate("F_T", x, Q2, nucleon),
                self.evaluate("F_L" if self.target_mass else "F_L_scaling", x, Q2, nucleon))


class Blend(InclusiveModel):
    """Linear blend in W of two models over [W_lo, W_hi] (for smooth transitions).

    ``q = (1-t) q_low + t q_high`` with t = (W - W_lo)/(W_hi - W_lo), for every
    quantity q separately (so the blended R is not exactly sigma_L/sigma_T of the
    blended cross sections). Its domain is [W_lo, W_hi] intersected with the domains
    of *both* models. Place it before the two models in a :class:`Piecewise` list.
    """

    def __init__(self, low, high, W_lo, W_hi, name=None):
        self.low, self.high, self.W_lo, self.W_hi = low, high, W_lo, W_hi
        self.targets = tuple(t for t in low.targets if t in high.targets)
        self.name = name or f"blend({low.name}->{high.name})"

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return ((W >= self.W_lo) & (W <= self.W_hi) & (Q2 >= 0)
                & self.low.domain(W, Q2, target) & self.high.domain(W, Q2, target))

    def sigma_T(self, W, Q2, target):
        W = np.asarray(W, float)
        return self.evaluate("sigma_T", W, Q2, target)

    def evaluate(self, quantity, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        t = np.clip((W - self.W_lo) / (self.W_hi - self.W_lo), 0.0, 1.0)
        return ((1 - t) * self.low.evaluate(quantity, W, Q2, target)
                + t * self.high.evaluate(quantity, W, Q2, target))


class Piecewise(InclusiveModel):
    """Ordered list of models; each point is evaluated with the first model whose domain contains it.

    Parameters
    ----------
    models : list of InclusiveModel
    outside : {"nan", "zero", "raise"}
        What to return where no model applies.
    """

    def __init__(self, models, outside="nan", name="piecewise"):
        if outside not in ("nan", "zero", "raise"):
            raise ValueError("outside must be 'nan', 'zero' or 'raise'")
        self.models = list(models)
        self.outside = outside
        self.name = name
        self.targets = tuple(sorted({t for m in self.models for t in m.targets}))

    def assign(self, W, Q2, target):
        """Index (into ``models``) of the model used at each point; -1 where none applies."""
        t = _norm_target(target)
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        which = np.full(W.shape, -1)
        for i, m in enumerate(self.models):
            if t not in m.targets:
                continue
            free = which < 0
            if not np.any(free):
                break
            which[free & m.domain(W, Q2, t)] = i
        return which

    def _evaluate(self, quantity, W, Q2, target):
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {sorted(QUANTITIES)}, got {quantity!r}")
        t = _norm_target(target)
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        which = self.assign(W, Q2, t)
        if np.any(which < 0) and self.outside == "raise":
            bad = np.argwhere(which < 0)[0]
            raise ValueError(f"no photoabsorption model for target {t!r} at "
                             f"W={W[tuple(bad)]:.4f}, Q2={Q2[tuple(bad)]:.4g}")
        out = np.full(W.shape, np.nan if self.outside == "nan" else 0.0)
        for i, m in enumerate(self.models):
            sel = which == i
            if np.any(sel):
                out[sel] = m.evaluate(quantity, W[sel], Q2[sel], t)
        return out, which

    def evaluate(self, quantity, W, Q2, target):
        return self._evaluate(quantity, W, Q2, target)[0]

    def sigma_T(self, W, Q2, target):
        return self._evaluate("sigma_T", W, Q2, target)[0]

    def F1(self, W, Q2, target):
        return self._evaluate("F1", W, Q2, target)[0]

    def F1F2(self, W, Q2, target):
        return self._evaluate("F1", W, Q2, target)[0], self._evaluate("F2", W, Q2, target)[0]

    def domain(self, W, Q2, target):
        return self.assign(W, Q2, target) >= 0

    def describe(self):
        """Human-readable list of the pieces, in priority order."""
        lines = []
        for i, m in enumerate(self.models):
            rng = ""
            if isinstance(m, ChristyBosted):
                rng = "  " + ", ".join(f"{t}: W in {m.W_range[t]}, Q2 <= {m.Q2_max[t]}" for t in m.targets)
            elif isinstance(m, ChristyBosted2021):
                rng = f"  W^2 <= {m.W2_max}, Q2 <= {m.Q2_max}, W >= {m.W_min}  ** PRELIMINARY **"
            lines.append(f"{i}: {m.name} [{', '.join(m.targets)}]{rng}")
        return "\n".join(lines)


def default_photoabsorption(binned=False, outside="nan", fit="2009"):
    """Christy-Bosted in its fit range, CB extrapolated down to threshold, zero below threshold.

    The middle piece covers PION_THRESHOLD <= W < 1.1 GeV (p, n) and W < 1.2 GeV
    (d) with the Christy-Bosted functional forms, which have the right threshold
    behaviour by construction (and the deuteron fit's nucleon average was
    constrained there by MAID2007 pseudo-data), but no proton data below 1.1 GeV
    entered the proton fit. Replace or precede it with a dedicated low-W model
    when available.

    ``fit="2021"`` returns instead [ChristyBosted2021, BelowThreshold]: the
    **PRELIMINARY** 2021 fit (emits a PreliminaryFitWarning; ``binned`` is ignored).
    """
    if fit == "2021":
        warnings.warn("default_photoabsorption(fit='2021') uses the PRELIMINARY, unpublished "
                      "F1F221 fit (SIMC F1F2IN21_v1.0.f, M. E. Christy). Check its status with "
                      "the author before using it in published work.",
                      cb21.PreliminaryFitWarning, stacklevel=2)
        return Piecewise([ChristyBosted2021(warn=False), BelowThreshold()], outside=outside,
                         name="Christy 2021 (PRELIMINARY)")
    if fit != "2009":
        raise ValueError("fit must be '2009' or '2021'")
    fit = ChristyBosted(binned=binned, name="Christy-Bosted")
    extrap = ChristyBosted(W_range={"p": (PION_THRESHOLD, 1.1), "n": (PION_THRESHOLD, 1.1),
                                    "d": (0.0, 1.2)},
                           binned=binned, name="Christy-Bosted (threshold extrapolation)")
    return Piecewise([fit, extrap, BelowThreshold()], outside=outside, name="default")


_DEFAULT = None


def inclusive_sf(W, target="p", quantity="sigma_T", Q2=0.0, model=None, return_source=False):
    """Photoabsorption cross sections or structure functions for target "p", "n" or "d".

    Parameters
    ----------
    W : array_like
        c.m. energy [GeV] (for "d": free-proton W, see module docstring).
    target : {"p", "n", "d"}
    quantity : str
        One of :data:`QUANTITIES`: "sigma_T", "sigma_L" [microbarn], "R", "F1", "F2",
        "F_T" (= 2F1), "F_L" (= (1+gamma^2)F2/x - 2F1 = R F_T), "F_L_scaling"
        (= F2/x - 2F1), "xF_L" (= (1+gamma^2)F2 - 2xF1). See the module docstring
        for the conventions; F_T, F_L follow arXiv:2603.23699 and arXiv:2603.23700.
    Q2 : array_like
        Photon virtuality [GeV^2], default 0 (real photons).
    model : InclusiveModel, optional
        Default: :func:`default_photoabsorption` (built once).
    return_source : bool
        Also return an array with the name of the model piece used per point
        ("" where none applied).
    """
    global _DEFAULT
    if model is None:
        if _DEFAULT is None:
            _DEFAULT = default_photoabsorption()
        model = _DEFAULT
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {sorted(QUANTITIES)}, got {quantity!r}")
    if not model.supports(target):
        raise ValueError(f"model {model.name!r} does not describe target {target!r}")
    if isinstance(model, Piecewise):
        val, which = model._evaluate(quantity, W, Q2, target)
        if return_source:
            names = np.array([m.name for m in model.models] + [""], dtype=object)
            return val, names[which]
        return val
    W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
    val = model.evaluate(quantity, W, Q2, _norm_target(target))
    val = np.where(model.domain(W, Q2, target), val, np.nan)
    if return_source:
        return val, np.where(np.isnan(val), "", model.name).astype(object)
    return val


#: alias: real photoproduction is the Q^2 = 0 case of :func:`inclusive_sf`
photoproduction = inclusive_sf
#: alias of :class:`InclusiveModel` (former name)
PhotoabsorptionModel = InclusiveModel


# ===========================================================================
# ready-made combinations
# ===========================================================================
def resonance_dis_model(dis=None, W_blend=(2.5, 3.0), fit="2009", dis_Q2_min=1.0, outside="nan"):
    """Christy-Bosted in the resonance region, a DIS provider above, linear blend in between.

    Pieces (first match wins), for p and n:

    1. :class:`Blend` from Christy-Bosted to the DIS provider over ``W_blend``
       (only where both are valid);
    2. the DIS provider (:class:`FromNucleonStructure`) for W >= W_blend[1], Q^2 >= ``dis_Q2_min``;
    3. the Christy-Bosted pieces of :func:`default_photoabsorption` (``fit`` = "2009" or
       the PRELIMINARY "2021"), including threshold extrapolation and zero below threshold;
    4. the DIS provider wherever none of the above applies (e.g. Q^2 above the
       Christy-Bosted range), Q^2 >= ``dis_Q2_min``.

    The deuteron is served by the Christy-Bosted pieces only. Not covered (NaN by
    default): W above the Christy-Bosted range at Q^2 < ``dis_Q2_min`` (e.g. real photons
    at W > 3 GeV) -- the place for a Regge-type model.

    Parameters
    ----------
    dis : NucleonStructure, optional
        Default: ``SLACStructure(R=0.18, target_mass=True)`` (arXiv:2603.23700's input).
    W_blend : (float, float)
        Transition window in W [GeV]; must lie inside the Christy-Bosted range
        (W <= 3.0 GeV for the 2009 neutron).
    """
    from .nucleon import SLACStructure
    if dis is None:
        dis = SLACStructure(R=0.18, target_mass=True)
    dis_model = FromNucleonStructure(dis, Q2_min=dis_Q2_min)
    dis_hi = FromNucleonStructure(dis, W_min=W_blend[1], Q2_min=dis_Q2_min, name=dis_model.name)
    if fit == "2021":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cb21.PreliminaryFitWarning)
            res = default_photoabsorption(fit="2021", outside=outside)
        warnings.warn("resonance_dis_model(fit='2021') uses the PRELIMINARY, unpublished F1F221 fit.",
                      cb21.PreliminaryFitWarning, stacklevel=2)
    else:
        res = default_photoabsorption(fit=fit, outside=outside)
    blend = Blend(res.models[0], dis_model, *W_blend)
    # blend; DIS above the window; Christy-Bosted pieces (fit range, threshold
    # extrapolation, zero below threshold); DIS wherever none of these applies
    # (e.g. Q^2 above the Christy-Bosted range)
    pieces = [blend, dis_hi] + list(res.models) + [dis_model]
    return Piecewise(pieces, outside=outside, name=f"resonance({res.models[0].name})+DIS({dis_model.name})")


class ChristyBostedStructure(AsNucleonStructure):
    """Unpolarized F1, F2 (and F_T, F_L) of p and n from a single Christy-Bosted fit, in (x, Q^2).

    ``AsNucleonStructure(ChristyBosted(...))``, i.e. W^2 = M^2 + Q^2 (1/x - 1) with the
    physical proton/neutron mass (as nucqcd ``NucleonStructure::getF_CB`` and the
    other providers; until 2026-11-19 this class used the fit's proton mass for both,
    which changes the neutron by up to 1-2% near threshold, i.e. at large x, and the
    proton by < 1e-5). Evaluated everywhere
    (extrapolated outside the fit range, as the FORTRAN does); ``in_range`` checks the
    range. For Christy-Bosted in the resonance region and a DIS fit above, use
    ``AsNucleonStructure(resonance_dis_model())`` instead.

    Proton: Christy & Bosted, PRC 81, 055213 (2010); neutron: 2(p+n)/2 - p with the
    nucleon average from Bosted & Christy, PRC 77, 065206 (2008); see
    :mod:`taggedpd.christy_bosted`. ``FT_FL`` returns F_[UU,T] = 2F1 and
    F_[UU,L] = R F_T from the fit's own R.

    Parameters
    ----------
    binned : bool
        Reproduce the FORTRAN's 1 MeV W binning in the neutron (default False: smooth).
    version : {"2009", "2021"}
        "2009" (default): the published fits. "2021": the **PRELIMINARY**, unpublished
        F1F221 fit of M. E. Christy (SIMC ``F1F2IN21_v1.0.f``, see
        :mod:`taggedpd.christy_bosted21`); emits a PreliminaryFitWarning.
    """

    def __init__(self, binned=False, version="2009"):
        if version not in ("2009", "2021"):
            raise ValueError("version must be '2009' or '2021'")
        if version == "2021":
            warnings.warn("ChristyBostedStructure(version='2021') uses the PRELIMINARY, "
                          "unpublished F1F221 fit (SIMC F1F2IN21_v1.0.f, M. E. Christy).",
                          cb21.PreliminaryFitWarning, stacklevel=2)
            model = ChristyBosted2021(warn=False)
        else:
            model = ChristyBosted(binned=binned)
        super().__init__(model, target_mass=True,
                         name="Christy-Bosted" if version == "2009" else "Christy 2021 (PRELIMINARY)")
        self.binned = binned
        self.version = version

    def in_range(self, x, Q2, nucleon="n"):
        W = self.W(x, Q2, nucleon)
        w2 = W * W
        Q2 = np.asarray(Q2, float)
        if self.version == "2021":
            return (W > 1.08) & (w2 < 20.0) & (Q2 >= 0) & (Q2 < 30.0)
        Wmax = 3.1 if _norm_target(nucleon) == "p" else 3.0
        return (W > 1.1) & (W < Wmax) & (Q2 >= 0) & (Q2 < 8.0)

    def F1F2R(self, x, Q2, nucleon="n"):
        """(F1, F2, R) from the fit; zero outside 0 < x < 1."""
        return (self.evaluate("F1", x, Q2, nucleon), self.evaluate("F2", x, Q2, nucleon),
                self.evaluate("R", x, Q2, nucleon))
