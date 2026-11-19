r"""Total (transverse) photoabsorption on the nucleon and the deuteron.

Main entry point::

    from taggedpd.photoabsorption import photoproduction
    sigma = photoproduction(W, "n")                     # sigma_T(gamma n -> X) [microbarn] at Q^2 = 0
    F1 = photoproduction(W, "p", quantity="F1", Q2=0.1)

``photoproduction`` evaluates a :class:`PhotoabsorptionModel`. The default one
(:func:`default_photoabsorption`) is a :class:`Piecewise` combination:

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
Subclass :class:`PhotoabsorptionModel`, implement ``sigma_T`` *or* ``F1`` (the
other follows from Hand's convention) and ``domain``, then put it in a
:class:`Piecewise` list. Earlier entries win where domains overlap, so a
dedicated low-W model or a Regge-type high-W fit is simply placed *before* the
Christy-Bosted entries::

    class MyRegge(PhotoabsorptionModel):
        name = "Regge"
        targets = ("p", "n")
        def domain(self, W, Q2, target):
            return (W >= 2.5) & (Q2 == 0)
        def sigma_T(self, W, Q2, target):
            s = W**2
            return X * s**eps + Y * s**(-eta)           # microbarn

    model = Piecewise([MyRegge(), *default_photoabsorption().models])
    photoproduction(W, "p", model=model)

A smooth transition between two pieces can be had by a model whose ``domain``
is the overlap region and whose ``sigma_T`` interpolates between the two (see
:class:`Blend`).

Conventions
-----------
* GeV, cross sections in **microbarn**.
* ``W`` is the photon-nucleon c.m. energy. For the deuteron, ``W`` is (as in the
  Christy-Bosted code) the invariant mass *calculated as if the photon scattered
  on a free proton at rest*, i.e. a label for the photon lab energy
  nu = (W^2 - M^2 + Q^2)/2M; use :func:`W_from_nu` / :func:`nu_from_W`.
* Deuteron results are **per nucleus** (divide by 2 for per nucleon).
* sigma_T = 4 pi^2 alpha F1/(M K) with Hand's flux K = (W^2 - M^2)/(2M) and M the
  proton mass of the fit, i.e. sigma_T = 8 pi^2 alpha F1/(W^2 - M^2). At
  Q^2 = 0, K equals the photon energy, so sigma_T is the real-photon total
  absorption cross section; sigma_L and F2 vanish there.
* The deuteron result is the *inelastic* (pion-production) part only:
  photodisintegration gamma d -> p n is not included.

Structure-function conventions (``quantity=``)
----------------------------------------------
Conventions for "F_T" and "F_L" differ between papers by factors of x and 2, so
every option is defined explicitly here. All are built from the model's own F1, F2
and the nucleon mass M that the model uses kinematically (:meth:`PhotoabsorptionModel.kin_mass`):
nu = (W^2 - M^2 + Q^2)/2M, x = Q^2/(2 M nu), gamma^2 = 4 M^2 x^2/Q^2 = Q^2/nu^2.

=============== ========================================== ==================================
quantity        definition                                  remarks
=============== ========================================== ==================================
"sigma_T"       transverse cross section [microbarn]         Hand flux K = (W^2 - M^2)/2M
"sigma_L"       R sigma_T [microbarn]
"R"             sigma_L/sigma_T = F_L/F_T
"F1"            sigma_T (W^2 - M^2)/(8 pi^2 alpha)           fits' native output
"F2"            F2                                           0 at Q^2 = 0
"F_T"           F_[UU,T] = 2 F1                              paper / Part I convention
"F_L"           F_[UU,L] = (1 + gamma^2) F2/x - 2 F1         exact (target-mass) relation;
                = R F_T                                      Part I inclusive limit
"F_L_scaling"   F2/x - 2 F1                                  scaling-limit relation
                = F_T (R - gamma^2)/(1 + gamma^2)            (gamma -> 0), paper Eq. (5.10)
"xF_L"          (1 + gamma^2) F2 - 2 x F1 = 2 x F1 R         the "F_L" of Christy-Bosted /
                = x F_[UU,L]                                 SIMC and of many DIS papers
=============== ========================================== ==================================

Notes:

* F_T and F_L follow the paper (dimension of F1). Elsewhere one also finds
  F_T = 2xF1 with F_L = (1+gamma^2)F2 - 2xF1 (= "xF_L" here), such that
  (1+gamma^2) F2 = F_T + F_L. That convention is x times the one used here.
* "F_L_scaling" is what the paper's Eq. (5.10) gives with F1, F2 as input; it
  equals F_L only when gamma^2 << 1. In the resonance region at low Q^2,
  gamma^2 = Q^2/nu^2 is *not* small (e.g. ~0.5 at Q^2 = 1, W = 1.5), and
  F_L_scaling can even be negative where gamma^2 > R. Use "F_L" for the
  physical longitudinal structure function.
* At Q^2 = 0 all F_L-type quantities, F2, R and sigma_L are returned as exactly 0
  (their limits; x = 0 there). Models that only provide sigma_T/F1 (e.g. a
  user-supplied Regge form) give F2- and F_L-type quantities only at Q^2 = 0 and
  raise ``NotImplementedError`` otherwise.
* Deuteron: per nucleus; x and gamma use the fit's nucleon mass (2009: 0.93828;
  2021: the average nucleon mass), i.e. x is per nucleon, 0 < x < 2.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import christy_bosted as cb
from . import christy_bosted21 as cb21

__all__ = ["PhotoabsorptionModel", "ChristyBosted", "ChristyBosted2021", "BelowThreshold",
           "Piecewise", "Blend", "default_photoabsorption", "photoproduction", "W_from_nu",
           "nu_from_W", "PION_THRESHOLD", "QUANTITIES"]

_M = cb.MP_CB
#: free-nucleon pion-production threshold W = m_p + m_pi0 [GeV], as in the fits
PION_THRESHOLD = cb.MP_CB + 0.135


def W_from_nu(nu, Q2=0.0, mass=_M):
    """W for photon lab energy ``nu`` on a nucleon of ``mass`` at rest: W^2 = M^2 + 2 M nu - Q^2."""
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
class PhotoabsorptionModel:
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
        """Transverse cross section [microbarn]."""
        if type(self).F1 is PhotoabsorptionModel.F1:
            raise NotImplementedError(f"{type(self).__name__} implements neither sigma_T nor F1")
        W = np.asarray(W, float)
        return cb.sigmaT_from_F1(self.F1(W, Q2, target), W * W, self.flux_mass)

    def F1(self, W, Q2, target):
        """F1 (per nucleon for p, n; per nucleus for d)."""
        if type(self).sigma_T is PhotoabsorptionModel.sigma_T:
            raise NotImplementedError(f"{type(self).__name__} implements neither sigma_T nor F1")
        W = np.asarray(W, float)
        return cb.F1_from_sigmaT(self.sigma_T(W, Q2, target), W * W, self.flux_mass)

    def kin_mass(self, target):
        """Nucleon mass M the model uses for nu = (W^2 - M^2 + Q^2)/2M, x = Q^2/2M nu, gamma^2 = Q^2/nu^2."""
        return _M

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
            x = Q2 / (2.0 * M * nu)
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


#: quantities :func:`photoproduction` / :meth:`PhotoabsorptionModel.evaluate` can return.
#: See the module docstring, "Structure-function conventions".
QUANTITIES = {
    "sigma_T": "transverse cross section [microbarn] (Hand flux)",
    "sigma_L": "longitudinal cross section [microbarn] = R sigma_T",
    "R": "sigma_L / sigma_T = F_L / F_T",
    "F1": "F1 = sigma_T (W^2 - M^2) / (8 pi^2 alpha)",
    "F2": "F2",
    "F_T": "F_[UU,T] = 2 F1  (paper / Part I convention)",
    "F_L": "F_[UU,L] = (1 + gamma^2) F2 / x - 2 F1 = R F_T  (exact, with gamma^2 = 4M^2x^2/Q^2)",
    "F_L_scaling": "F2 / x - 2 F1  (scaling-limit relation gamma -> 0, paper Eq. (5.10))",
    "xF_L": "x F_[UU,L] = (1 + gamma^2) F2 - 2 x F1 = 2 x F1 R  (the 'F_L' of Christy-Bosted / SIMC, "
            "of many DIS papers)",
}


# ===========================================================================
# Christy-Bosted
# ===========================================================================
class ChristyBosted(PhotoabsorptionModel):
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
        (smooth). True: 1 MeV W bins as in the FORTRAN (and the physics-code C++),
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
        W = np.asarray(W, float)
        return cb.f1f2in09(_norm_target(target), Q2, W * W, binned=self.binned)[0]

    def F1F2R(self, W, Q2, target):
        """(F1, F2, R) as returned by F1F2IN09."""
        W = np.asarray(W, float)
        return cb.f1f2in09(_norm_target(target), Q2, W * W, binned=self.binned)

    def F1F2(self, W, Q2, target):
        return self.F1F2R(W, Q2, target)[:2]

    def kin_mass(self, target):
        """F1F2IN09 builds F2 with nu from PM = 0.93828 GeV, for p, n and d."""
        return cb._PM


class ChristyBosted2021(PhotoabsorptionModel):
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
        W = np.asarray(W, float)
        return cb21.f1f2in21(_norm_target(target), Q2, W * W)[0]

    def F1F2(self, W, Q2, target):
        W = np.asarray(W, float)
        return cb21.f1f2in21(_norm_target(target), Q2, W * W)

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


class BelowThreshold(PhotoabsorptionModel):
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


class Blend(PhotoabsorptionModel):
    """Linear blend in W of two models over [W_lo, W_hi] (for smooth transitions).

    ``q = (1-t) q_low + t q_high`` with t = (W - W_lo)/(W_hi - W_lo), for every
    quantity q separately (so the blended R is not exactly sigma_L/sigma_T of the
    blended cross sections). Place it before the two models in a :class:`Piecewise` list.
    """

    def __init__(self, low, high, W_lo, W_hi, name=None):
        self.low, self.high, self.W_lo, self.W_hi = low, high, W_lo, W_hi
        self.targets = tuple(t for t in low.targets if t in high.targets)
        self.name = name or f"blend({low.name}->{high.name})"

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= self.W_lo) & (W <= self.W_hi) & (Q2 >= 0)

    def sigma_T(self, W, Q2, target):
        W = np.asarray(W, float)
        return self.evaluate("sigma_T", W, Q2, target)

    def evaluate(self, quantity, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        t = np.clip((W - self.W_lo) / (self.W_hi - self.W_lo), 0.0, 1.0)
        return ((1 - t) * self.low.evaluate(quantity, W, Q2, target)
                + t * self.high.evaluate(quantity, W, Q2, target))


class Piecewise(PhotoabsorptionModel):
    """Ordered list of models; each point is evaluated with the first model whose domain contains it.

    Parameters
    ----------
    models : list of PhotoabsorptionModel
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


def photoproduction(W, target="p", quantity="sigma_T", Q2=0.0, model=None, return_source=False):
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
        for the conventions; F_T, F_L follow the paper (Part I).
    Q2 : array_like
        Photon virtuality [GeV^2], default 0 (real photons).
    model : PhotoabsorptionModel, optional
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
