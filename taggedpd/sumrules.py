r"""Integrals of the LF distributions over the spectator momentum (paper Sec. IV, App. A).

The integration measure :math:`\int d\alpha_p/\alpha_p\, d^2p_{pT}` is converted to the
pn c.m. momentum (Eq. (3.21)):

.. math::
    \int_0^2 \frac{d\alpha_p}{\alpha_p} d^2 p_{pT}\, P(\alpha_p, p_{pT})
    = \int \frac{d^3k}{E(k)}\, (2-\alpha_p)\, P = \int \frac{d^3k}{E}\, N(\mathbf k),

and done with Gauss-Legendre quadrature in |k| (mapped to [0, k_max]) and
cos(theta_k); the integrand does not depend on the azimuth of k because the
transverse basis rotates with k_T (Eq. (4.29)). These routines are used by the
test-suite (sum rules (4.23)-(4.25), (A10)-(A29), Table III) and the notebook.
"""

from __future__ import annotations

import numpy as np

from .constants import M_NUCLEON

__all__ = ["integrate_distribution", "sum_rule_table"]


def _k_grid(k_max=3.0, n_k=400, n_c=64):
    # map u in [-1,1] -> k = k_max * ((1+u)/2)^2 to concentrate points at small k
    u, wu = np.polynomial.legendre.leggauss(n_k)
    t = 0.5 * (1 + u)
    k = k_max * t**2
    wk = wu * 0.5 * 2 * k_max * t
    c, wc = np.polynomial.legendre.leggauss(n_c)
    return k[:, None], wk[:, None], c[None, :], wc[None, :]


def alpha_pT_from_k(k, cos_t, m=M_NUCLEON):
    """Inverse of Eq. (3.20): (alpha_p, pT, E) from |k| and cos(theta_k)."""
    E = np.sqrt(k * k + m * m)
    alpha_p = 1.0 + k * cos_t / E
    pT = k * np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    return alpha_p, pT, E


def integrate_distribution(lf, name, weight=None, k_max=3.0, n_k=400, n_c=64):
    r"""\int d alpha_p/alpha_p d^2 p_pT  w(alpha_p) P_name(alpha_p, pT).

    Parameters
    ----------
    lf : LFDistributions
    name : str
        attribute of :class:`NeutronDistributions` (e.g. ``"UU"``, ``"SL_SL"``).
    weight : callable(alpha_p) or None
        extra weight, e.g. ``lambda a: 2 - a`` for the momentum sum rule (4.24).
    """
    k, wk, c, wc = _k_grid(k_max, n_k, n_c)
    a, pT, E = alpha_pT_from_k(k, c)
    P = getattr(lf(a, pT), name)
    integrand = (2.0 - a) * P / E * 2 * np.pi * k * k     # d^3k/E N, phi_k done
    if weight is not None:
        integrand = integrand * weight(a)
    return float(np.sum(wk * wc * integrand))


def sum_rule_table(lf):
    """The integrals of Table III (exact column) plus the unpolarized sum rules."""
    I = lambda name, w=None: integrate_distribution(lf, name, w)  # noqa: E731
    omega2 = lf.wf.norm_D
    return {
        "norm (4.23)": I("UU"),
        "momentum (4.24)": I("UU", lambda a: 2 - a),
        "TLL (4.25a)": I("TLL_U"),
        "TLL (4.25b)": I("TLL_U", lambda a: 2 - a),
        "TLT (4.25)": I("TLT_U"),
        "TTT (4.25)": I("TTT_U"),
        "omega2 (A7)": omega2,
        "I[SL] (A1)": I("SL_SL"),
        "1-3/2 omega2": 1 - 1.5 * omega2,
        "delta_par (A20)": 1 - I("ST_ST_par"),
        "delta_perp (A22)": 1 - I("ST_ST_perp"),
        "eps[SL] (A11)": I("ST_SL"),
        "eps[ST] (A28)": I("SL_ST"),
        "(A27)": 0.5 * (integrate_distribution(lf, "ST_ST_par", lambda a: 1 / (2 - a))
                        + integrate_distribution(lf, "ST_ST_perp", lambda a: 1 / (2 - a))),
    }
