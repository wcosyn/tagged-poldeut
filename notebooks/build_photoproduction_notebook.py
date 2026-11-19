"""Builds photoproduction.ipynb (plain nbformat-4 JSON, no jupyter needed).

Run:  python notebooks/build_photoproduction_notebook.py         -> writes notebooks/photoproduction.ipynb
      python notebooks/build_photoproduction_notebook.py --run   -> also executes all code cells
                                                                   (Agg backend), PNGs to notebooks/figures/
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s):
    cells.append(("code", s.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Photoabsorption on p, n and d — Christy–Bosted

This notebook shows the Python port of the Christy–Bosted resonance-region fits
(`taggedpd.christy_bosted`) and the photoproduction interface built on it
(`taggedpd.photoabsorption`).

**Original work — please cite:**

* M. E. Christy and P. E. Bosted, *Empirical fit to precision inclusive electron-proton cross sections
  in the resonance region*, Phys. Rev. C **81**, 055213 (2010), arXiv:0712.3731 — proton,
  $0 \le Q^2 < 8$ GeV², $1.1 < W < 3.1$ GeV, $\sigma_T$ and $\sigma_L$ separately, includes real-photon data.
* P. E. Bosted and M. E. Christy, *Empirical fit to inelastic electron-deuteron and electron-neutron
  resonance region transverse cross sections*, Phys. Rev. C **77**, 065206 (2008), arXiv:0711.0159 —
  deuteron and neutron, $0 \le Q^2 < 10$ GeV², $1.2 < W < 3$ GeV (MAID2007 pseudo-data below 1.2 GeV).
* Code: `F1F209.f` by P. E. Bosted (JLab); the copy ported here is `physics-code/Tools/F1F209_test.f`.
  The port reproduces it to machine precision (see `tests/test_christy_bosted.py`).

Conventions: GeV, cross sections in µb, $\sigma_T = 8\pi^2\alpha\,F_1/(W^2-M^2)$ (Hand), deuteron
**per nucleus** with $W$ computed as for a free proton (a label for the photon energy $\nu$).
""")

code(r"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))          # use the repo checkout
from taggedpd import christy_bosted as cb
from taggedpd.photoabsorption import (photoproduction, default_photoabsorption, Piecewise, Blend,
                                      ChristyBosted, PhotoabsorptionModel, nu_from_W, W_from_nu,
                                      PION_THRESHOLD)

plt.rcParams.update({'axes.labelsize': 13, 'legend.fontsize': 10, 'figure.dpi': 90,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
FIGDIR = os.environ.get('TAGGEDPD_FIGDIR')
def save(fig, name):
    if FIGDIR:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, name + '.png'), bbox_inches='tight')

model = default_photoabsorption()
print(model.describe())
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. How the neutron (and the deuteron) are obtained

There is **no separate neutron fit**. Bosted & Christy parametrize the *free-nucleon average*
$\bar F_1 = (F_{1p}+F_{1n})/2$ (routine `RESMODD`: same Breit–Wigner + background form as the proton
fit, own parameters). The deuteron is then modelled as (`RESD`)

$$F_1^d(W^2,Q^2)\big/2 \;=\; \sum_i w_i\,\bar F_1\big(W'^2_i,Q^2\big) \;+\; F_1^{\rm MEC}(W^2,Q^2),$$

a Fermi smearing over the deuteron momentum distribution (tabulated bins $p_z$, $\langle p^2\rangle$,
weights; the struck nucleon is off shell with energy $M_d-\sqrt{M^2+p^2}$), plus an empirical,
purely transverse term filling the dip between the quasi-elastic peak and the $\Delta$. The parameters
of $\bar F_1$ and the MEC term are fitted to inclusive deuteron data. The neutron follows as

$$F_{1n} = 2\bar F_1 - F_{1p}^{\rm Christy}.$$

Consequences: the neutron carries the uncertainties of both fits and the model dependence of the
unfolding (smearing, MEC); the deuteron fit is transverse only, so $R_n$ (and $R_d$) are taken from
the proton (Fermi-smeared for the deuteron); at $Q^2=0$ only $F_1 \leftrightarrow \sigma_T$ survives.
The deuteron result is the **inelastic** part: photodisintegration $\gamma d\to pn$ is not included.
""")

code(r"""
W = np.linspace(1.08, 3.0, 1200)
w2 = W**2
p = cb.f1f2in09('p', 0.0, w2)[0]
avg = cb.resmodd(w2, 0.0, binned=False)
n = 2 * avg - p
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(W, p, label=r'$F_{1p}$ (Christy)')
ax[0].plot(W, avg, label=r'$\bar F_1=(F_{1p}+F_{1n})/2$ (deuteron fit)')
ax[0].plot(W, n, label=r'$F_{1n}=2\bar F_1-F_{1p}$')
ax[0].set_xlabel('$W$ [GeV]'); ax[0].set_ylabel('$F_1$ at $Q^2=0$'); ax[0].legend()
ax[1].plot(W, n / p)
ax[1].axhline(1, c='0.6', lw=0.8)
ax[1].set_xlabel('$W$ [GeV]'); ax[1].set_ylabel(r'$\sigma_T^{\gamma n}/\sigma_T^{\gamma p}$'); ax[1].set_ylim(0.4, 1.6)
fig.tight_layout(); save(fig, 'pp_neutron_construction')
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Real-photon total cross sections

$\sigma_T(\gamma p)$, $\sigma_T(\gamma n)$ and $\sigma_T(\gamma d)/2$ versus $W$ and versus the photon lab
energy. The shaded band marks where the default model uses the Christy–Bosted forms *outside* the
published fit range (between the pion threshold and $W=1.1$ GeV for p, n; $W<1.2$ GeV for d). As a
sanity check, the $\Delta(1232)$ peak of $\gamma p$ is about 0.5 mb (PDG). The deuteron, plotted per
nucleon, shows the Fermi-smeared resonances and the strength below the free-nucleon pion threshold
(smearing + MEC).
""")

code(r"""
W = np.linspace(1.0, 3.2, 1500)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for tgt, lab, fac in (('p', r'$\gamma p$', 1), ('n', r'$\gamma n$', 1), ('d', r'$\gamma d$ / 2', 0.5)):
    sig, src = photoproduction(W, tgt, return_source=True)
    ax[0].plot(W, fac * sig, label=lab)
    nu = nu_from_W(W)
    ax[1].plot(nu, fac * sig, label=lab)
ax[0].axvspan(PION_THRESHOLD, 1.1, color='0.85', zorder=0, label='threshold extrapolation (p, n)')
ax[0].set_xlabel('$W$ [GeV]'); ax[0].set_ylabel(r'$\sigma_T$ [$\mu$b]'); ax[0].legend()
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$E_\gamma$ [GeV]'); ax[1].set_ylabel(r'$\sigma_T$ [$\mu$b]')
ax[1].legend()
fig.tight_layout(); save(fig, 'pp_sigmaT')

Wd = np.linspace(1.15, 1.35, 2001)
sp = photoproduction(Wd, 'p')
print(f'gamma p: Delta peak sigma_T = {sp.max():.0f} microbarn at W = {Wd[np.argmax(sp)]:.3f} GeV')
""")

code(r"""
# deuteron vs the sum of free nucleons: Fermi smearing and the MEC term
W = np.linspace(1.0, 2.2, 1000)
sd = photoproduction(W, 'd')
spn = photoproduction(W, 'p') + photoproduction(W, 'n')
mec = cb.sigmaT_from_F1(2 * cb.mec_deuteron(0.0, W**2), W**2)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(W, sd, label=r'$\gamma d$ (smeared $\bar F_1$ + MEC)')
ax.plot(W, spn, '--', label=r'$\gamma p + \gamma n$ (free)')
ax.plot(W, mec, ':', label='MEC term alone')
ax.set_xlabel('$W$ (free-proton kinematics) [GeV]'); ax.set_ylabel(r'$\sigma_T$ [$\mu$b] per nucleus'); ax.legend()
fig.tight_layout(); save(fig, 'pp_deuteron_vs_free')
""")

md(r"""
**Caveats visible above (properties of the fits, not of the port):**

* *Deuteron below $W\approx1.2$ GeV at $Q^2=0$.* The strength there is dominated by the empirical MEC
  term (~300 µb near the free-nucleon threshold). In the FORTRAN its $Q^2$ dependence is
  $1/(1+\max(0.3,Q^2)/a)^{b}$, i.e. it is frozen at its $Q^2=0.3$ GeV² value for real photons, and the
  region lies below the deuteron fit's $W$ range. Treat it as unconstrained; it is labelled
  "threshold extrapolation" by the default model. Photodisintegration is not included at all.
* *$\sigma_T(\gamma n)/\sigma_T(\gamma p)\approx1.15$ at the $\Delta$ peak.* The $\Delta$ is isovector, and
  data extracted from the deuteron give $\gamma n\approx\gamma p$ there, so this ratio reflects the
  unfolding (smearing + MEC) in the neutron extraction. Compare with neutron data or MAID before
  relying on the neutron at the 10% level near the $\Delta$.
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. $F_1$ for p, n, d at several $Q^2$

$F_1(W,Q^2)$; deuteron per nucleus divided by 2. At finite $Q^2$, $F_2$ and $R$ are also available
(`cb.f1f2in09`, or `ChristyBostedStructure` as a `NucleonStructure` provider in $(x,Q^2)$).
""")

code(r"""
W = np.linspace(1.08, 3.0, 800)
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, tgt, title, fac in zip(axs, 'pnd', ('proton', 'neutron', 'deuteron / 2'), (1, 1, 0.5)):
    for Q2 in (0.0, 0.1, 0.5, 1.0, 2.0, 4.0):
        ax.plot(W, fac * photoproduction(W, tgt, 'F1', Q2=Q2), label=f'$Q^2$ = {Q2:g}')
    ax.set_title(title); ax.set_xlabel('$W$ [GeV]')
axs[0].set_ylabel('$F_1$'); axs[0].legend()
fig.tight_layout(); save(fig, 'pp_F1_Q2')
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. The approach to $Q^2 = 0$

The limit $Q^2\to0$ is continuous, but some resonance heights vary on a very small $Q^2$ scale. For
example, the proton fit's second resonance (S11(1535)) has a height factor
$1 + 6823\,Q^2/(1+33521\,Q^2)$, which rises from 1 to 1.2 between $Q^2 = 0$ and $Q^2\approx 10^{-3}$
GeV² (and enters squared). This is how the fit reconciles the real-photon point with the
electroproduction data; it is a feature of the fit, not of the port. Practical consequence: for
quasi-real photons ($Q^2\sim 10^{-5}$–$10^{-3}$ GeV², e.g. small-angle electroproduction) the result is not
the $Q^2=0$ value.
""")

code(r"""
Q2 = np.concatenate([[0.0], np.geomspace(1e-8, 2.0, 200)])
fig, ax = plt.subplots(figsize=(7, 4))
for Wv in (1.232, 1.5, 1.7, 2.5):
    for tgt, ls in (('p', '-'), ('n', '--')):
        s = photoproduction(np.full_like(Q2, Wv), tgt, Q2=Q2)
        ax.plot(Q2[1:], s[1:] / s[0], ls, label=f'{tgt}, W = {Wv}')
ax.set_xscale('log'); ax.set_xlabel('$Q^2$ [GeV$^2$]'); ax.set_ylabel(r'$\sigma_T(Q^2)/\sigma_T(0)$')
ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); save(fig, 'pp_smallQ2')
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Binned (FORTRAN) vs smooth evaluation

`RESMODD` evaluates its Breit–Wigner shapes from a table in 1 MeV bins of $W$. The port reproduces this
with `binned=True` (used in the regression tests); the photoproduction interface uses the smooth
evaluation (`binned=False`) by default. The binning produces a staircase in $\bar F_1$ and hence in the
neutron: ~1% above $W=1.2$ GeV and up to tens of percent within ~25 MeV of the pion threshold, where $F_1$
itself is small. The physics-code C++ (which links the FORTRAN) has the staircase.
""")

code(r"""
W = np.linspace(PION_THRESHOLD + 1e-4, 1.25, 3000)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for binned, lab in ((True, 'binned (FORTRAN)'), (False, 'smooth')):
    ax[0].plot(W, photoproduction(W, 'n', model=default_photoabsorption(binned=binned)), label=lab)
ax[0].set_xlabel('$W$ [GeV]'); ax[0].set_ylabel(r'$\sigma_T(\gamma n)$ [$\mu$b]'); ax[0].legend()
W = np.linspace(1.08, 3.0, 6000)
a = cb.f1f2in09('n', 0.0, W**2, binned=True)[0]; b = cb.f1f2in09('n', 0.0, W**2, binned=False)[0]
ax[1].plot(W, a / b - 1)
ax[1].set_xlabel('$W$ [GeV]'); ax[1].set_ylabel('neutron: binned/smooth - 1'); ax[1].set_ylim(-0.1, 0.1)
fig.tight_layout(); save(fig, 'pp_binning')
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Adding other parametrizations

`photoproduction` evaluates an ordered `Piecewise` list: each point uses the first model whose
`domain` contains it, and `return_source=True` reports which piece was used. A new model implements
`domain` and either `sigma_T` or `F1`. Below, a **toy** Regge-like form
$\sigma = X s^{\epsilon} + Y s^{-\eta}$ (illustrative numbers, *not a fit*) is placed in front of the
default pieces for $W > 2.5$ GeV, with a linear `Blend` over $2.3 < W < 2.5$ GeV for a continuous
transition. A dedicated low-$W$ model would be inserted the same way.
""")

code(r"""
class ToyRegge(PhotoabsorptionModel):
    name = 'toy Regge'
    targets = ('p', 'n')
    def domain(self, W, Q2, target):
        W, Q2 = np.broadcast_arrays(np.asarray(W, float), np.asarray(Q2, float))
        return (W >= 2.5) & (Q2 == 0)
    def sigma_T(self, W, Q2, target):
        s = np.asarray(W, float)**2
        return 67.7 * s**0.0808 + 129.0 * s**-0.4525      # microbarn, illustrative

toy = ToyRegge()
cbfit = ChristyBosted()
ext = Piecewise([Blend(cbfit, toy, 2.3, 2.5), toy, *default_photoabsorption().models])
print(ext.describe())

W = np.linspace(1.0, 6.0, 2000)
sig, src = photoproduction(W, 'p', model=ext, return_source=True)
fig, ax = plt.subplots(figsize=(7, 4))
for name in dict.fromkeys(src):
    m = src == name
    ax.plot(W[m], sig[m], '.', ms=2, label=name or 'none')
ax.set_xlabel('$W$ [GeV]'); ax.set_ylabel(r'$\sigma_T(\gamma p)$ [$\mu$b]'); ax.set_yscale('log'); ax.legend(markerscale=6)
fig.tight_layout(); save(fig, 'pp_piecewise')
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Speed

Everything is vectorized. p and n cost ~1 µs per point; the deuteron (Fermi smearing over 20 or 200
bins) ~30 µs. For event generation at fixed $Q^2$, tabulate once in $W$ and interpolate.
""")

code(r"""
import time
rng = np.random.default_rng(0)
Wr = rng.uniform(1.1, 3.0, 200_000)
for tgt in 'pnd':
    t0 = time.time(); photoproduction(Wr, tgt); dt = time.time() - t0
    print(f'{tgt}: {dt / Wr.size * 1e6:.2f} us/point')
""")


# ---------------------------------------------------------------------------
def build():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {},
                "source": [line + "\n" for line in src.split("\n")]}
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    with open(os.path.join(HERE, "photoproduction.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)


def run():
    import matplotlib
    matplotlib.use("Agg")
    os.environ.setdefault("TAGGEDPD_FIGDIR", os.path.join(HERE, "figures"))
    os.chdir(HERE)
    g = {"__name__": "__main__"}
    for kind, src in cells:
        if kind == "code":
            exec(compile(src, "<notebook>", "exec"), g)


if __name__ == "__main__":
    build()
    if "--run" in sys.argv:
        run()
