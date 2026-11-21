"""Builds paper_figures.ipynb (plain nbformat-4 JSON, no jupyter needed).

Run:  python notebooks/build_notebook.py         -> writes notebooks/paper_figures.ipynb
      python notebooks/build_notebook.py --run   -> also executes all code cells as a
                                                    script (Agg backend), saving PNGs to
                                                    notebooks/figures/ (CI-style check)
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
# Tagged DIS on the polarized deuteron — figures of arXiv:2603.23700

Reproduces the figures of W. Cosyn & C. Weiss, *Semi-inclusive deep-inelastic scattering on a polarized spin-1 target. II. Deuteron and spectator nucleon tagging*, Phys. Rev. C **114**, 025202 (2026), [arXiv:2603.23700](https://arxiv.org/abs/2603.23700) (JLAB-THY-26-4661)
with the standalone `taggedpd` package, and cross-checks against output of the
`physics-code` C++ implementation (`Poldeut`, `NucleonStructure`).

Inputs as in arXiv:2603.23700 (Sec. VI A): AV18 deuteron wave function (CD-Bonn for the model
comparison), SLAC/Bodek $F_{2n}$ with $R = 0.18$, polarized $g_{1n}$ from GRSV2000 LO
(the C++ code uses GRSV2000; the text of arXiv:2603.23700 quotes DSSV09) and $g_{2n}$ from the
Wandzura–Wilczek relation.

Figure and equation numbers refer to arXiv:2603.23700; the companion paper is
W. Cosyn & C. Weiss, *Semi-inclusive deep-inelastic scattering on a polarized spin-1 target. I. Cross section and spin observables*, Phys. Rev. C **114**, 025201 (2026), [arXiv:2603.23699](https://arxiv.org/abs/2603.23699) (JLAB-THY-26-4663).
""")

code(r"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.insert(0, os.path.abspath('..'))          # use the repo checkout
import taggedpd
from taggedpd import (DeuteronWavefunction, LFDistributions, TaggedDIS, Kinematics,
                      Polarization, default_structure)
from taggedpd.sumrules import sum_rule_table

plt.rcParams.update({'axes.labelsize': 14, 'legend.fontsize': 11, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'figure.dpi': 90, 'font.family': 'serif',
                     'mathtext.fontset': 'cm'})
FIGDIR = os.environ.get('TAGGEDPD_FIGDIR')         # set to save PNGs
def save(fig, name):
    if FIGDIR:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, name + '.png'), bbox_inches='tight')

# optional: directory with the C++ reference output (only used for overlay points)
CPP_RESULTS = os.path.expanduser('~/Calculations/poldeut/results')
REF = os.path.abspath(os.path.join('..', 'tests', 'reference'))

lf = LFDistributions('AV18')
model = TaggedDIS('AV18')                 # tagged proton, active neutron
model_cdb = TaggedDIS('CDBonn', structure=model.structure)
print('taggedpd', taggedpd.__version__, '|', model.lf.wf, '|', model.structure.name)
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Wave function normalization and sum rules

The radial wave functions are normalized as $4\pi\int dk\,k^2/E\,[f_0^2+f_2^2] = 1$ (Eq. 3.23),
with $f_L = \sqrt{E/4\pi}\,U_L$ (Eq. 3.24). For Yukawa parametrizations this integral is known in
closed form, $\sum_{ij}C_iC_j/(m_i+m_j)$, which the test-suite compares with numerical quadrature.
The D wave is positive by convention ($W(k)>0$ at small $k$).

The table below shows the integrals of the LF distributions (Eqs. 4.23–4.25 and Table III, "exact"
column of arXiv:2603.23700: $I_{[S_L]} = 0.911$, $\delta^\parallel = 0.125$, $\delta^\perp = 0.0512$,
$\varepsilon_{[S_L]} = -0.0568$, $\varepsilon_{[S_T]} = 0.0550$, $\omega_2 = 0.0576$ for AV18).

*Note:* $P_{[T_{TT},U]}$ alone, and $(2-\alpha_p)P_{[T_{LT},U]}$, do **not** integrate to zero; they only drop
out of the sum rules through the $\cos(2\phi_p-2\phi_{T_T})$, $\cos(\phi_p-\phi_{T_L})$ factors of Eq. (4.22).
""")

code(r"""
for name in ('AV18', 'CDBonn', 'Paris'):
    wf = DeuteronWavefunction(name)
    print(f'{name:7s} norm (exact) = {wf.norm:.7f}   P_D = omega_2 = {wf.norm_D:.5f}')
print()
tabs = {w: sum_rule_table(LFDistributions(w)) for w in ('AV18', 'CDBonn', 'Paris')}
print(f"{'integral':22s}" + ''.join(f'{w:>10s}' for w in tabs))
for key in tabs['AV18']:
    print(f'{key:22s}' + ''.join(f'{tabs[w][key]:10.4f}' for w in tabs))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
k = np.linspace(0, 1.0, 600)
for name, ls in (('AV18', '-'), ('CDBonn', '--'), ('Paris', ':')):
    wf = DeuteronWavefunction(name)
    f0, f2 = wf.f0f2(k)
    ax[0].plot(k, k * k * f0, 'C0' + ls, label=f'$k^2 f_0$ {name}')
    ax[0].plot(k, k * k * f2, 'C1' + ls, label=f'$k^2 f_2$ {name}')
    ax[1].plot(k, f2 / f0, ls, label=name)
ax[0].set_xlabel('$k$ [GeV]'); ax[0].legend(ncol=2, fontsize=8); ax[0].axhline(0, c='0.6', lw=0.5)
ax[1].axhline(np.sqrt(2), c='0.5', ls='--', lw=1); ax[1].text(0.02, 1.5, r'$f_2/f_0=\sqrt{2}$')
ax[1].set_ylim(-2, 4); ax[1].set_xlim(0, 0.6); ax[1].set_xlabel('$k$ [GeV]'); ax[1].set_ylabel('$f_2/f_0$'); ax[1].legend()
fig.tight_layout(); save(fig, 'wavefunctions')
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Neutron LF distributions (Sec. IV)

### Fig. 3 / Fig. 4 — $P_{[U,U]}$
Points: C++ `Poldeut::getLFdistributions_new` output (if available).
""")

code(r"""
pT = np.linspace(0, 0.6, 301)
cpp = None
try:
    cpp = np.loadtxt(os.path.join(REF, 'lf_dens_AV18.txt.gz'))
except OSError:
    pass

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, a in enumerate((1.0, 1.1, 1.2)):
    ax[0].semilogy(pT, lf(a, pT).UU, f'C{i}', label=rf'$\alpha_p={a}$')
    if cpp is not None:
        m = np.isclose(cpp[:, 0], a)
        ax[0].semilogy(cpp[m, 1][::4], cpp[m, 2][::4], f'C{i}o', ms=3, mfc='none')
ax[0].set_ylim(1e-2, 2e3); ax[0].set_xlabel(r'$p_{pT}$ [GeV]'); ax[0].set_ylabel(r'$P_{[U,U]}$ [GeV$^{-2}$]')
ax[0].legend(title='lines: taggedpd, o: C++'); ax[0].set_title('Fig. 3')

pT_f = np.linspace(0, 0.6, 1201)
for a in (1.0, 1.1, 1.2, 1.3, 1.4):
    w = 2 * np.pi * pT_f * lf(a, pT_f).UU
    ax[1].plot(pT_f, w / np.trapezoid(w, pT_f), label=rf'$\alpha_p={a}$')
ax[1].set_xlabel(r'$p_{pT}$ [GeV]'); ax[1].set_ylabel(r'normalized $2\pi p_{pT}P_{[U,U]}$ [GeV$^{-1}$]')
ax[1].legend(); ax[1].set_title('Fig. 4')
fig.tight_layout(); save(fig, 'fig03_04')
""")

md(r"""
### Fig. 5 — unpolarized and tensor-polarized distributions
Normalized tensor distributions of Eq. (4.19) (the C++ output differs by factors
$\sqrt{3/2}$, $\sqrt2$, $1/\sqrt2$, see `densities.py`).
""")

code(r"""
A, PT = np.meshgrid(np.linspace(0.5, 1.5, 401), np.linspace(0, 0.6, 301))
P = lf(A, PT)
fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, sharey=True)
c = ax[0, 0].contourf(A, PT, P.UU, levels=np.logspace(-4, 4, 17), norm=colors.LogNorm(), cmap='viridis')
fig.colorbar(c, ax=ax[0, 0])
for axx, data, lab in ((ax[0, 1], P.TLL_U, r'$P_{[T_{LL},U]}$'), (ax[1, 0], P.TLT_U, r'$P_{[T_{LT},U]}$'),
                       (ax[1, 1], P.TTT_U, r'$P_{[T_{TT},U]}$')):
    from matplotlib.ticker import MaxNLocator
    lev = MaxNLocator(12).tick_values(np.nanmin(data), np.nanmax(data))   # full range, as in arXiv:2603.23700
    lim = np.max(np.abs(lev)) * 2.0                                       # lighter colours, zero = white
    c = axx.contourf(A, PT, data, levels=lev, cmap='RdBu_r', norm=colors.TwoSlopeNorm(0, -lim, lim))
    fig.colorbar(c, ax=axx); axx.text(0.97, 0.93, lab + r' [GeV$^{-2}$]', transform=axx.transAxes, ha='right', va='top')
ax[0, 0].text(0.97, 0.93, r'$P_{[U,U]}$ [GeV$^{-2}$]', transform=ax[0, 0].transAxes, ha='right', va='top', color='w')
for a_ in ax[1]: a_.set_xlabel(r'$\alpha_p$')
for a_ in ax[:, 0]: a_.set_ylabel(r'$p_{pT}$ [GeV]')
fig.suptitle('Fig. 5'); fig.tight_layout(); save(fig, 'fig05')
""")

md(r"""
### Fig. 6 / Fig. 7 — polarized neutron distributions (absolute values on log scale)
""")

code(r"""
pT = np.linspace(0, 0.6, 601)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, col in ((1.0, 'C0'), (1.2, 'C1')):
    P = lf(a, pT)
    ax[0].semilogy(pT, P.UU, col + '-', label=rf'$P_{{[U,U]}}$, $\alpha_p={a}$')
    ax[0].semilogy(pT, np.abs(P.SL_SL), col + '--', label=r'$|P_{[S_L,S_L]}|$')
    ax[0].semilogy(pT, np.abs(P.ST_ST_par), col + '-.', label=r'$|P^\parallel_{[S_T,S_T]}|$')
ax[0].set_ylim(1e-3, 2e3); ax[0].legend(fontsize=8, ncol=2); ax[0].set_title('Fig. 6')
for axx, a in ((ax[1], 1.0), (ax[2], 1.2)):
    P = lf(a, pT)
    axx.semilogy(pT, np.abs(P.SL_SL), 'C0-', label=r'$P_{[S_L,S_L]}$')
    axx.semilogy(pT, np.abs(P.SL_ST), 'C0--', label=r'$P_{[S_L,S_T]}$')
    axx.semilogy(pT, np.abs(P.ST_ST_par), 'C1-', label=r'$P^\parallel_{[S_T,S_T]}$')
    axx.semilogy(pT, np.abs(P.ST_SL), 'C1--', label=r'$P_{[S_T,S_L]}$')
    axx.set_ylim(1e-3, 2e3); axx.set_title(rf'Fig. 7, $\alpha_p={a}$ (absolute values)'); axx.legend(fontsize=9)
for a_ in ax: a_.set_xlabel(r'$p_{pT}$ [GeV]')
fig.tight_layout(); save(fig, 'fig06_07')
""")

md(r"""
### Fig. 8 — neutron polarization probabilities (Eqs. 4.48, 4.49)
""")

code(r"""
A, PT = np.meshgrid(np.linspace(0.4, 1.6, 241), np.linspace(0, 0.6, 241))
P = lf(A, PT)
fig, ax = plt.subplots(2, 2, figsize=(12, 6.2), sharex=True, sharey=True)
lev = np.linspace(0, 1, 11)
for axx, R, lab in ((ax[0, 0], P.R_LL, r'$R_{[L+,L+]}$'), (ax[0, 1], P.R_TL, r'$R_{[T+,L+]}$'),
                    (ax[1, 0], P.R_LT, r'$R_{[L+,T+]}$'), (ax[1, 1], P.R_TT, r'$R_{[T+,T+]}$')):
    c = axx.contourf(A, PT, R, levels=lev)
    axx.text(0.97, 0.93, lab, transform=axx.transAxes, ha='right', va='top',
             bbox=dict(fc='white', ec='white', alpha=0.4))
fig.colorbar(c, ax=ax)
for a_ in ax[1]: a_.set_xlabel(r'$\alpha_p$')
for a_ in ax[:, 0]: a_.set_ylabel(r'$p_{pT}$ [GeV]')
fig.suptitle('Fig. 8'); save(fig, 'fig08')
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Observables (Sec. VI)

### Fig. 9 / Fig. 15 — unpolarized differential cross section
$d\sigma_U/dx\,dQ^2\,(d\alpha_p/\alpha_p)\,d^2p_{pT}$ [nb/GeV$^4$], $\phi_p$-averaged, at fixed $x_n = 0.05$,
$Q^2 = 6$ GeV$^2$, and $s = 800$ GeV$^2$ **per nucleon** (5×41 GeV; this is the convention of arXiv:2603.23700's
plot script, which gives the quoted $\epsilon = 0.98$; `Kinematics.from_s(..., per_nucleon=True)`).

**Discrepancy with the arXiv:2603.23700 figure.** The plot scripts for Figs. 9 and 15
(`tagdeut_unpolcross_log.py`, `tagdeut_unpolcross_ndep.py`) convert to nb with `0.197*0.197*10E07`;
`10E07` = $10^8$ instead of $10^7$, so the published curves are **10× too large**. Thin dotted:
script emulation (×10). The model-dependence ratio of Fig. 15 is unaffected.
""")

code(r"""
pT = np.linspace(0, 0.6, 301)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xn, Q2, s = 0.05, 6.0, 800.0
for a in (1.0, 1.1, 1.2):
    kin = Kinematics.from_s(xn * (2 - a), Q2, s, dis_limit=True, per_nucleon=True)
    ds = model.dsigma(kin, a, pT, phi_averaged=True)
    l, = ax[0].semilogy(pT, ds, label=rf'$\alpha_p={a}$, $\epsilon={float(kin.epsilon):.3f}$')
    ax[0].semilogy(pT, 10 * ds, ':', c=l.get_color(), lw=0.8)
ax[0].set_ylim(5, 2e6); ax[0].legend(); ax[0].set_title(r'Fig. 9: $x_n=0.05$, $Q^2=6$ GeV$^2$, $s_{eN}=800$ GeV$^2$ (dotted: $\times10$)')
ax[0].set_xlabel(r'$p_{pT}$ [GeV]'); ax[0].set_ylabel(r'$d\sigma_U/dx\,dQ^2\,\frac{d\alpha_p}{\alpha_p}d^2p_T$ [nb/GeV$^4$]')

# Fig. 15: wave-function model dependence
a = 1.0
kin = Kinematics.from_s(xn * (2 - a), Q2, s, dis_limit=True, per_nucleon=True)
ax[1].semilogy(pT, model.dsigma(kin, a, pT, phi_averaged=True), 'k-', label='AV18')
ax[1].semilogy(pT, model_cdb.dsigma(kin, a, pT, phi_averaged=True), 'k--', label='CDBonn')
ax[1].set_ylim(1, 1e4); ax[1].legend(); ax[1].set_title(r'Fig. 15, $\alpha_p = 1$')
ax[1].set_xlabel(r'$p_{pT}$ [GeV]')
fig.tight_layout(); save(fig, 'fig09_15')
r = model_cdb.dsigma(kin, a, 0.5, phi_averaged=True) / model.dsigma(kin, a, 0.5, phi_averaged=True)
print(f'CDBonn/AV18 at pT = 0.5 GeV: {float(r):.2f}   (arXiv:2603.23700: ~0.6)')
""")

md(r"""
### Fig. 10 — azimuthal modulation of the unpolarized cross section (Eq. 6.2)
$\alpha_p = 1$, $x_n = 0.1$, $Q^2 = 2$ GeV$^2$, $\epsilon = 0.8$.
""")

code(r"""
phi = np.linspace(0, 2 * np.pi, 361)
a, xn, Q2, eps = 1.0, 0.1, 2.0, 0.8
kin = Kinematics.from_epsilon(xn * (2 - a), Q2, eps)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i, pt in enumerate((0.0, 0.2, 0.4, 0.6)):
    sf = model.structure_functions(kin.x, Q2, a, pt)
    full = model.assemble(sf, eps, phi) / model.assemble(sf, eps, 0, phi_averaged=True)
    sf1 = dict(sf, UU_cos_2phi=0.0 * sf['UU_cos_2phi'])
    only1 = model.assemble(sf1, eps, phi) / model.assemble(sf1, eps, 0, phi_averaged=True)
    ax.plot(np.degrees(phi), full, f'C{i}-', label=rf'$p_{{pT}}={pt}$ GeV')
    ax.plot(np.degrees(phi), only1, f'C{i}--')
ax.set_xlabel(r'$\phi_p$ [deg]'); ax.set_ylabel(r'$\mathcal{F}_U(\phi_p)/\langle\mathcal{F}_U\rangle$')
ax.set_xticks([0, 90, 180, 270, 360]); ax.legend(title=r'solid: all, dashed: $\cos\phi_p$ only', fontsize=9)
ax.set_title('Fig. 10'); fig.tight_layout(); save(fig, 'fig10')
""")

md(r"""
### Fig. 11 / Fig. 16 — vector-polarized structure function ratio $F_{[LS_L]D}/\Sigma_\parallel F_D$ (Eq. 6.5)
$x_n = 0.1$, $Q^2 = 2$ GeV$^2$, $\epsilon = 0.8$. The cross-check line uses `vector_asymmetry`
(two-state double-spin asymmetry from pure deuteron states, arXiv:2603.23699) divided by $D_{\parallel[S_L]} = \sqrt{1-\epsilon^2}$.
""")

code(r"""
pT = np.linspace(0, 0.6, 301)
xn, Q2, eps = 0.1, 2.0, 0.8
def ratio_par(m, a, pt):
    kin = Kinematics.from_epsilon(xn * (2 - a), Q2, eps)
    sf = m.structure_functions(kin.x, Q2, a, pt)
    sig = sf['UU_T'] + eps * sf['UU_L'] + (sf['UTLL_T'] + eps * sf['UTLL_L']) / 3
    return sf['LSL'] / sig, kin

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
styles = {0.6: 'b:', 0.8: 'b--', 1.0: 'k-', 1.2: 'r--', 1.4: 'r:'}
for a, st in styles.items():
    r, kin = ratio_par(model, a, pT)
    ax[0].plot(pT, r, st, label=rf'$\alpha_p={a}$')
r, kin = ratio_par(model, 1.2, pT)
AV = model.vector_asymmetry(kin, 1.2, pT[::15], 'parallel') / np.sqrt(1 - eps**2)
ax[0].plot(pT[::15], AV, 'ro', ms=4, mfc='none', label=r'$A^V_\parallel/D_{\parallel[S_L]}$, $\alpha_p=1.2$')
ax[0].axhline(0, c='0.5', ls='--', lw=0.8); ax[0].set_ylim(-0.12, 0.12); ax[0].legend(fontsize=9)
ax[0].set_title('Fig. 11'); ax[0].set_ylabel(r'$F_{[L,S_L]D}/\Sigma_\parallel F_D$')
ax[1].plot(pT, ratio_par(model, 1.0, pT)[0], 'k-', label='AV18')
ax[1].plot(pT, ratio_par(model_cdb, 1.0, pT)[0], 'k--', label='CDBonn')
ax[1].axhline(0, c='0.5', ls='--', lw=0.8); ax[1].set_ylim(-0.12, 0.12); ax[1].legend(); ax[1].set_title(r'Fig. 16, $\alpha_p=1$')
for a_ in ax: a_.set_xlabel(r'$p_{pT}$ [GeV]')
fig.tight_layout(); save(fig, 'fig11_16')
""")

md(r"""
### Fig. 12 — transversely polarized ratio $F^{\cos\phi_S}_{[LS_T]D}/\Sigma_\perp F_D$ (Eq. 6.10b)

**Discrepancy with the draft figure.** The plot scripts used for Fig. 12
(`tagdeut_ALL_ALT*.py`) computed $\gamma_n$ as `2*x_n*0.93/Q2` instead of $2x_nm/Q$
(Eq. 5.22), and took $g_1+g_2$ from the then C++ Wandzura–Wilczek routine, a 101-point Riemann
sum that overestimates $|g_1+g_2|$ by 7.5% at $x = 0.1$. Together these made the draft curves
~30% too small. Both are fixed now (scripts corrected, C++ WW fixed in physics-code `e9d1e6a`,
`neutron_SF*.txt` regenerated); the corrected scripts reproduce the solid curves. Solid: correct ($\gamma_n = 2x_nm/Q$, accurate WW). Thin dotted: emulating the
script (`ww='cpp'`, $\gamma_n \to 2x_n\cdot0.93/Q^2$), which reproduces arXiv:2603.23700's values
(e.g. ≈0.0075 at $\alpha_p = 1$, $p_{pT} = 0$).
""")

code(r"""
model_cpp_ww = TaggedDIS('AV18', structure=default_structure(ww='cpp'))
def ratio_perp(m, a, pt, emulate_script=False):
    kin = Kinematics.from_epsilon(xn * (2 - a), Q2, eps)
    sf = m.structure_functions(kin.x, Q2, a, pt)
    sig = sf['UU_T'] + eps * sf['UU_L'] - (sf['UTLL_T'] + eps * sf['UTLL_L']) / 6
    r = sf['LST_cos_phiS'] / sig
    if emulate_script:     # gamma_n -> 2 x_n 0.93 / Q2 instead of 2 x_n m / Q
        r = r * (2 * xn * 0.93 / Q2) / sf['nucleon']['gamma_n']
    return r

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for a, st in styles.items():
    ax.plot(pT, ratio_perp(model, a, pT), st, label=rf'$\alpha_p={a}$')
    ax.plot(pT, ratio_perp(model_cpp_ww, a, pT, True), st[0] + ':', lw=0.7)
ax.axhline(0, c='0.5', ls='--', lw=0.8); ax.legend(fontsize=9); ax.set_xlabel(r'$p_{pT}$ [GeV]')
ax.set_ylabel(r'$F^{\cos\phi_S}_{[L,S_T]D}/\Sigma_\perp F_D$'); ax.set_title('Fig. 12 (thin: original-script emulation)')
fig.tight_layout(); save(fig, 'fig12')
print('alpha_p=1, pT=0:  correct', float(ratio_perp(model, 1.0, 0.0)),
      '  script emulation', float(ratio_perp(model_cpp_ww, 1.0, 0.0, True)))
""")

md(r"""
### Fig. 13 / Fig. 17 — tensor-polarized asymmetry $A^T_\parallel$ (Eq. 6.12)
Computed from pure deuteron states $\Lambda = \pm1, 0$ along the beam (arXiv:2603.23699 Eq. AT_gen) in the DIS
limit — it reduces to $\sqrt{2/3}\,P_{[T_{LL},U]}/P_{[U,U]}$ (checked in the tests). Range $[-2, 1]$.
""")

code(r"""
kin = Kinematics.from_epsilon(0.1, 2.0, 0.8)    # A^T_par is independent of x, Q2, eps in the IA
pT = np.linspace(0, 0.6, 301); al = np.linspace(0.4, 1.6, 481)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for a, st in ((1.0, 'b:'), (1.15, 'b--'), (1.3, 'k-'), (1.45, 'r--'), (1.6, 'r:')):
    ax[0].plot(pT, model.tensor_asymmetry(kin, a, pT), st, label=rf'$\alpha_p={a}$')
ax[0].set_xlabel(r'$p_{pT}$ [GeV]'); ax[0].set_title('Fig. 13 (upper)')
for i, pt in enumerate((0.4, 0.3, 0.2, 0.1, 0.0)):
    ax[1].plot(al, model.tensor_asymmetry(kin, al, pt), f'C{i}', label=rf'$p_{{pT}}={pt}$ GeV')
ax[1].set_xlabel(r'$\alpha_p$'); ax[1].set_title('Fig. 13 (lower)')
for pt, col in ((0.3, 'r'), (0.0, 'k')):
    ax[2].plot(al, model.tensor_asymmetry(kin, al, pt), col + '-', label=rf'AV18, $p_T={pt}$')
    ax[2].plot(al, model_cdb.tensor_asymmetry(kin, al, pt), col + '--', label='CDBonn')
ax[2].set_xlabel(r'$\alpha_p$'); ax[2].set_title('Fig. 17')
for a_ in ax:
    a_.set_ylim(-2.05, 1.05); a_.axhline(0, c='0.5', ls='--', lw=0.8); a_.legend(fontsize=9); a_.set_ylabel(r'$A^T_\parallel$')
fig.tight_layout(); save(fig, 'fig13_17')
""")

md(r"""
### Fig. 14 — $\phi_p$-dependent tensor asymmetry $A^T_\perp$
Deuteron polarized perpendicular to the beam at $\psi_N = 0$, $\alpha_p = 1$, $x_n = 0.1$, $Q^2 = 2$ GeV$^2$,
$\epsilon = 0.8$; the denominator is the full $\phi_p$-dependent unpolarized cross section.
Dashed: without the $\cos3\phi_p$, $\cos4\phi_p$ structures in the numerator.
""")

code(r"""
phi = np.linspace(0, 2 * np.pi, 361)
a, xn, Q2, eps = 1.0, 0.1, 2.0, 0.8
kin = Kinematics.from_epsilon(xn * (2 - a), Q2, eps)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i, pt in enumerate((0.0, 0.1, 0.2, 0.3, 0.4)):
    AT = model.tensor_asymmetry(kin, a, pt, 'perp', psi_N=0.0, phi_p=phi, phi_averaged=False)
    # numerator without cos3phi and cos4phi terms: Lambda-combination, theta_N = pi/2, phi_N = 0
    sf = model.structure_functions(kin.x, Q2, a, pt)
    FU = model.assemble(sf, eps, phi)
    num = (-(sf['UTLL_T'] + eps * sf['UTLL_L'] + np.sqrt(2 * eps * (1 + eps)) * np.cos(phi) * sf['UTLL_cos_phi']
             + eps * np.cos(2 * phi) * sf['UTLL_cos_2phi'])
           + 3 * (np.cos(2 * phi) * (sf['UTTT_T'] + eps * sf['UTTT_L'])
                  + np.sqrt(2 * eps * (1 + eps)) * np.cos(phi) * sf['UTTT_tw3'] + eps * sf['UTTT_tw4']))
    ax.plot(np.degrees(phi), AT, f'C{i}-', label=rf'$p_{{pT}}={pt}$ GeV')
    ax.plot(np.degrees(phi), num / (3 * FU), f'C{i}--')
ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlabel(r'$\phi_p$ [deg]'); ax.set_ylabel(r'$A^T_\perp$')
ax.set_ylim(-2.05, 1.2); ax.legend(fontsize=9); ax.set_title('Fig. 14')
fig.tight_layout(); save(fig, 'fig14')
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Normalization check of $\sigma_{\rm red}$ (Eq. 2.22)

$\int d\Gamma_p\,\sigma_{{\rm red},D} \approx F_{2n} - (1-\epsilon)F_{Ln}$ "up to nuclear binding effects":
integrate the $\phi_p$-averaged reduced cross section over the spectator momentum at fixed $x$, $Q^2$, $\epsilon$
and compare with the free-neutron reduced cross section at $x_n = x$ (the integral smears over $x_n = x/(2-\alpha_p)$).
""")

code(r"""
from taggedpd.sumrules import _k_grid, alpha_pT_from_k
from taggedpd.constants import TWO_2PI3
k, wk, c, wc = _k_grid(2.0, 200, 48)
a_, pt_, E_ = alpha_pT_from_k(k, c)
for x in (0.05, 0.1, 0.2, 0.4):
    kin = Kinematics.from_epsilon(x, 5.0, 0.7)
    sred = model.reduced_cross_section(kin, a_, pt_, phi_averaged=True)
    integral = np.sum(wk * wc * sred * (2 - a_) * 2 * np.pi * k * k / E_) / TWO_2PI3
    FT, FL = model.structure.FT_FL(x, 5.0, 'n'); F2 = model.structure.F1F2(x, 5.0, 'n')[1]
    print(f'x = {x:4.2f}:  int dGamma sigma_red = {integral:.4f}   F2n - (1-eps) x FLn = {F2 - 0.3 * x * FL:.4f}')
""")

md(r"""
## 5. Minimal Monte Carlo usage

Everything is vectorized; a few $\mu$s per event. Example: weights for events sampled uniformly in
$(x, Q^2, \alpha_p, p_{pT}, \phi_p)$ for a longitudinally (beam-parallel) polarized deuteron with
vector polarization $P_z = 0.8$, tensor polarization $P_{zz} = 0.6$, and electron helicity $+1/2$.
""")

code(r"""
import time
rng = np.random.default_rng(1)
N = 200_000
x = rng.uniform(0.02, 0.6, N); Q2 = rng.uniform(2, 40, N)
a = rng.uniform(0.7, 1.3, N); pt = rng.uniform(0, 0.4, N); ph = rng.uniform(0, 2 * np.pi, N)
t0 = time.time()
kin = Kinematics.from_s(x, Q2, s=4 * 18 * 110 * 2)          # e-D s for 18 GeV e on 110 GeV/A D
pol = Polarization.beam(kin, 'parallel', Pz=0.8, Pzz=0.6)
w = model.dsigma(kin, a, pt, ph, lambda_e=+0.5, pol=pol)     # nb/GeV^4 per dx dQ2 dalpha/alpha d^2pT
print(f'{N} weights in {time.time() - t0:.2f} s; physical fraction {kin.physical.mean():.2f}; mean weight {w.mean():.3g} nb/GeV^4')
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
    with open(os.path.join(HERE, "paper_figures.ipynb"), "w") as f:
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
