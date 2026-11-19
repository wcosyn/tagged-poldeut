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
* Section 7 compares with a **PRELIMINARY, unpublished** 2021 fit by M. E. Christy (F1F221 v1.0
  prerelease 1, from the JLab SIMC package). Do not use it for publications without checking its
  status with the author.

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
## 7. Comparison with the PRELIMINARY 2021 fit (F1F221)

> **PRELIMINARY — not for publication without checking with the author.** The 2021 fit is the
> unpublished "F1F221 v1.0 prerelease 1" of M. E. Christy (April 2022), distributed with the JLab
> Hall C simulation package SIMC as `F1F2IN21_v1.0.f`
> ([github.com/JeffersonLab/simc_gfortran](https://github.com/JeffersonLab/simc_gfortran)). The
> deuteron/neutron part is attributed there to M. E. Christy, N. Kalantarians, J. Ethier and
> W. Melnitchouk, "to be published", and uses inclusive deuteron data, BONuS tagged n/d data and
> photoproduction at $Q^2=0$. The port is `taggedpd.christy_bosted21`, which agrees with the
> FORTRAN to rounding; it is opt-in (`default_photoabsorption(fit="2021")`) and emits a
> `PreliminaryFitWarning`. The 2009 fits remain the default and the citable reference.

Main differences: the neutron has its own fit, with its own $\sigma_T$ and $\sigma_L$, instead of
$2\bar F_1 - F_{1p}$; the deuteron is p + n smeared with CD-Bonn light-cone distributions, with no
MEC term; and the range extends to $W^2<20$, $Q^2<30$ GeV².
""")

code(r"""
import warnings
from taggedpd import christy_bosted21 as cb21
from taggedpd.christy_bosted21 import PreliminaryFitWarning
with warnings.catch_warnings():
    warnings.simplefilter('ignore', PreliminaryFitWarning)   # we know: preliminary
    m21 = default_photoabsorption(fit='2021')
m09 = default_photoabsorption()
print(m21.describe())

W = np.linspace(1.0, 3.2, 1200)
fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
for k, (tgt, title) in enumerate((('p', r'$\gamma p$'), ('n', r'$\gamma n$'), ('d', r'$\gamma d$ (per nucleus)'))):
    s09 = photoproduction(W, tgt, model=m09)
    s21 = photoproduction(W, tgt, model=m21)
    axs[0, k].plot(W, s09, label='2009 (PRC 81 055213 / PRC 77 065206)')
    axs[0, k].plot(W, s21, '--', label='2021 F1F221 (PRELIMINARY)')
    axs[0, k].set_title(title); axs[0, k].legend(fontsize=8)
    with np.errstate(divide='ignore', invalid='ignore'):
        axs[1, k].plot(W, s21 / s09, 'k')
    axs[1, k].axhline(1, c='0.6', lw=0.8); axs[1, k].set_ylim(0.5, 1.5)
    axs[1, k].set_xlabel('$W$ [GeV]')
axs[0, 0].set_ylabel(r'$\sigma_T$ [$\mu$b], $Q^2=0$'); axs[1, 0].set_ylabel('2021 / 2009')
fig.suptitle('Real-photon cross sections: 2009 vs 2021 (PRELIMINARY)', y=1.0)
fig.tight_layout(); save(fig, 'pp_cmp_2009_2021_sigmaT')
""")

code(r"""
# neutron/proton ratio and deuteron vs free p + n, both fits
W = np.linspace(1.08, 3.0, 1000)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for m, lab, ls in ((m09, '2009', '-'), (m21, '2021 (PRELIMINARY)', '--')):
    sp, sn, sd = (photoproduction(W, t, model=m) for t in 'pnd')
    ax[0].plot(W, sn / sp, ls, label=lab)
    ax[1].plot(W, sd / (sp + sn), ls, label=lab)
ax[0].axhline(1, c='0.6', lw=0.8); ax[0].set_ylim(0.4, 2.0)
ax[0].set_xlabel('$W$ [GeV]'); ax[0].set_ylabel(r'$\sigma_T(\gamma n)/\sigma_T(\gamma p)$'); ax[0].legend()
ax[1].axhline(1, c='0.6', lw=0.8); ax[1].set_ylim(0.5, 2.0)
ax[1].set_xlabel('$W$ [GeV]'); ax[1].set_ylabel(r'$\sigma_T(\gamma d)/[\sigma_T(\gamma p)+\sigma_T(\gamma n)]$'); ax[1].legend()
fig.tight_layout(); save(fig, 'pp_cmp_ratios')
Wd = np.array([1.21, 1.232]); print('sigma_n/sigma_p at the Delta: 2009', np.round(photoproduction(Wd, 'n') / photoproduction(Wd, 'p'), 3),
                                  ' 2021', np.round(photoproduction(Wd, 'n', model=m21) / photoproduction(Wd, 'p', model=m21), 3))
""")

code(r"""
# F1 at finite Q^2, both fits
W = np.linspace(1.08, 3.0, 600)
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, tgt, title, fac in zip(axs, 'pnd', ('proton', 'neutron', 'deuteron / 2'), (1, 1, 0.5)):
    for i, Q2 in enumerate((0.1, 0.5, 2.0, 5.0)):
        ax.plot(W, fac * photoproduction(W, tgt, 'F1', Q2=Q2, model=m09), c=f'C{i}', label=f'$Q^2$={Q2:g} 2009')
        ax.plot(W, fac * photoproduction(W, tgt, 'F1', Q2=Q2, model=m21), '--', c=f'C{i}', label=f'$Q^2$={Q2:g} 2021 (prelim.)')
    ax.set_title(title); ax.set_xlabel('$W$ [GeV]')
axs[0].set_ylabel('$F_1$'); axs[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); save(fig, 'pp_cmp_F1_Q2')
""")

md(r"""
### Checks of the 2021 port

* Regression: all outputs ($\sigma_{T,L}$ of p and n, and $F_1$, $F_2$ of p, n, d) agree with the FORTRAN to
  rounding (`tests/test_christy_bosted21.py`).
* The smearing functions are normalized: $\int f_{11}\,dy = \int f_{22}\,dy = 1$ at $\gamma = 1$.
* The original code has quirks that are reproduced on purpose and documented in the module. The most
  visible one: below $Q^2=0.05$ GeV² the deuteron uses a separate smearing routine whose integrand is
  $F_1(z)$ instead of $F_1(z)/y$, which makes the deuteron $F_1$ **discontinuous at $Q^2=0.05$** (left
  plot). Also, `GETFY` reads the table one $y$ bin too high.
* The 2021 range extends to $W^2 < 20$ GeV² (right plot; the 2009 fits stop at $W\approx3$ GeV).
""")

code(r"""
W = np.array([1.08, 1.12, 1.232, 1.5, 2.0])
Q2 = np.linspace(0.02, 0.08, 241)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for Wv in W:
    F1 = cb21.f1f2in21('d', Q2, np.full_like(Q2, Wv**2))[0]
    ax[0].plot(Q2, F1 / F1[np.argmin(abs(Q2 - 0.06))], label=f'W = {Wv}')
ax[0].axvline(0.05, c='0.6', ls=':')
ax[0].set_xlabel('$Q^2$ [GeV$^2$]'); ax[0].set_ylabel(r'$F_1^d(Q^2)/F_1^d(0.06)$  (2021, PRELIMINARY)'); ax[0].legend(fontsize=8)
Wl = np.linspace(1.0, 4.5, 1500)
for tgt in 'pn':
    ax[1].plot(Wl, photoproduction(Wl, tgt, model=m21), label=f'{tgt}, 2021 (PRELIMINARY)')
    ax[1].plot(Wl, photoproduction(Wl, tgt, model=m09), ':', label=f'{tgt}, 2009')
ax[1].set_xlabel('$W$ [GeV]'); ax[1].set_ylabel(r'$\sigma_T$ [$\mu$b]'); ax[1].set_yscale('log'); ax[1].legend(fontsize=8)
fig.tight_layout(); save(fig, 'pp_2021_checks')
""")

md(r"""
**What the comparison shows** (properties of the fits, reproduced faithfully by the ports):

* **Proton:** the two fits agree to within ~5% across the resonance region; they differ only near
  threshold, where the cross section is small.
* **Neutron:** differences of ±20% in the resonance region. At the $\Delta$ the 2021 fit gives
  $\sigma_n/\sigma_p\approx0.95$ (2009: 1.15). Near threshold the 2021 neutron is about twice the
  2009 one. At the $\eta$ threshold ($W\approx1.49$ GeV) the 2021 neutron has a cusp: its
  non-resonant $\eta$ term $(1-x'_\eta)^{t_1}$ has $t_1\approx0.03$ at $Q^2=0$, so it switches on
  almost like a step.
* **Deuteron:** the 2021 deuteron has no strength below the free-nucleon pion threshold (no MEC term
  in the inelastic part), whereas the 2009 deuteron gets ~150–300 µb there from its MEC term.
* **2021 deuteron, numerical noise:** besides the jump at $Q^2=0.05$, $F_1^d$ shows 1–5% steps as a
  function of $Q^2$, largest near threshold. They come from the fixed Simpson grid (48–76 nodes in
  $y$) crossing the 0.002 bins of the smearing table. This matters for MC weights at the few-% level
  near threshold.
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Speed

Everything is vectorized. With the 2009 fits, p and n cost ~1 µs per point and the deuteron (Fermi
smearing over 20 or 200 bins) ~30 µs. With the 2021 fit, the deuteron integrates over 48–76 $y$ nodes,
each evaluating both nucleon fits. For event generation at fixed $Q^2$, tabulate once in $W$ and
interpolate.
""")

code(r"""
import time
rng = np.random.default_rng(0)
Wr = rng.uniform(1.1, 3.0, 50_000)
for lab, m in (('2009', m09), ('2021 (prelim.)', m21)):
    for tgt in 'pnd':
        t0 = time.time(); photoproduction(Wr, tgt, model=m); dt = time.time() - t0
        print(f'{lab:15s} {tgt}: {dt / Wr.size * 1e6:.2f} us/point')
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
