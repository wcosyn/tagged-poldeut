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
"""

from __future__ import annotations

import numpy as np

from . import christy_bosted as cb

__all__ = ["PhotoabsorptionModel", "ChristyBosted", "BelowThreshold", "Piecewise", "Blend",
           "default_photoabsorption", "photoproduction", "W_from_nu", "nu_from_W",
           "PION_THRESHOLD"]

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

    def supports(self, target):
        return _norm_target(target) in self.targets

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"


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


class BelowThreshold(PhotoabsorptionModel):
    """sigma_T = 0 for a free nucleon below the pion-production threshold (Compton neglected)."""

    name = "below pion threshold"
    targets = ("p", "n")

    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W < PION_THRESHOLD) & (Q2 >= 0)

    def sigma_T(self, W, Q2, target):
        return np.zeros(np.broadcast(np.asarray(W), np.asarray(Q2)).shape)


class Blend(PhotoabsorptionModel):
    """Linear blend in W of two models over [W_lo, W_hi] (for smooth transitions).

    ``sigma_T = (1-t) low + t high`` with t = (W - W_lo)/(W_hi - W_lo). Place it
    before the two models in a :class:`Piecewise` list.
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
        t = np.clip((W - self.W_lo) / (self.W_hi - self.W_lo), 0.0, 1.0)
        return (1 - t) * self.low.sigma_T(W, Q2, target) + t * self.high.sigma_T(W, Q2, target)


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

    def _evaluate(self, method, W, Q2, target):
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
                out[sel] = getattr(m, method)(W[sel], Q2[sel], t)
        return out, which

    def sigma_T(self, W, Q2, target):
        return self._evaluate("sigma_T", W, Q2, target)[0]

    def F1(self, W, Q2, target):
        return self._evaluate("F1", W, Q2, target)[0]

    def domain(self, W, Q2, target):
        return self.assign(W, Q2, target) >= 0

    def describe(self):
        """Human-readable list of the pieces, in priority order."""
        lines = []
        for i, m in enumerate(self.models):
            rng = ""
            if isinstance(m, ChristyBosted):
                rng = "  " + ", ".join(f"{t}: W in {m.W_range[t]}, Q2 <= {m.Q2_max[t]}" for t in m.targets)
            lines.append(f"{i}: {m.name} [{', '.join(m.targets)}]{rng}")
        return "\n".join(lines)


def default_photoabsorption(binned=False, outside="nan"):
    """Christy-Bosted in its fit range, CB extrapolated down to threshold, zero below threshold.

    The middle piece covers PION_THRESHOLD <= W < 1.1 GeV (p, n) and W < 1.2 GeV
    (d) with the Christy-Bosted functional forms, which have the right threshold
    behaviour by construction (and the deuteron fit's nucleon average was
    constrained there by MAID2007 pseudo-data), but no proton data below 1.1 GeV
    entered the proton fit. Replace or precede it with a dedicated low-W model
    when available.
    """
    fit = ChristyBosted(binned=binned, name="Christy-Bosted")
    extrap = ChristyBosted(W_range={"p": (PION_THRESHOLD, 1.1), "n": (PION_THRESHOLD, 1.1),
                                    "d": (0.0, 1.2)},
                           binned=binned, name="Christy-Bosted (threshold extrapolation)")
    return Piecewise([fit, extrap, BelowThreshold()], outside=outside, name="default")


_DEFAULT = None


def photoproduction(W, target="p", quantity="sigma_T", Q2=0.0, model=None, return_source=False):
    """Photoabsorption sigma_T [microbarn] or F1 for target "p", "n" or "d".

    Parameters
    ----------
    W : array_like
        c.m. energy [GeV] (for "d": free-proton W, see module docstring).
    target : {"p", "n", "d"}
    quantity : {"sigma_T", "F1"}
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
    if quantity not in ("sigma_T", "F1"):
        raise ValueError("quantity must be 'sigma_T' or 'F1'")
    if not model.supports(target):
        raise ValueError(f"model {model.name!r} does not describe target {target!r}")
    if isinstance(model, Piecewise):
        val, which = model._evaluate(quantity, W, Q2, target)
        if return_source:
            names = np.array([m.name for m in model.models] + [""], dtype=object)
            return val, names[which]
        return val
    W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
    val = getattr(model, quantity)(W, Q2, _norm_target(target))
    val = np.where(model.domain(W, Q2, target), val, np.nan)
    if return_source:
        return val, np.where(np.isnan(val), "", model.name).astype(object)
    return val
