# taggedpd — polarized deuteron DIS with spectator tagging

A standalone, numpy-only Python implementation of the impulse-approximation cross section for

  e(λₑ) + D(S_D, T_D) → e′ + X + p (or n)

from W. Cosyn & C. Weiss, *Semi-inclusive deep-inelastic scattering on a polarized spin-1
target. II. Deuteron and spectator nucleon tagging* (JLAB-THY-26-4661, "the paper"), with
the cross-section decomposition of Part I (same authors). It is a port of the relevant
parts of `physics-code` (`Poldeut`, `TDeuteron`/`TYukawaPWF`, `NucleonStructure`), aimed
at use in an MC event generator. There is no numerical integration at run time: about
3 µs per event, vectorized.

Development history, validation details and open items: [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md).

## Install / test

```bash
pip install -e .            # runtime: numpy only
python -m pytest            # needs pytest + scipy
python notebooks/build_notebook.py --run   # regenerate notebook + PNGs in notebooks/figures/
python notebooks/build_photoproduction_notebook.py --run   # photoproduction / Christy-Bosted notebook
```

## Usage

```python
import numpy as np
from taggedpd import TaggedDIS, Kinematics, Polarization

model = TaggedDIS("AV18")                    # or "CDBonn", "Paris"; tagged proton (default)
kin = Kinematics.from_s(x=0.1, Q2=10.0, s=1600.0)     # s = e-D invariant [GeV^2]
#     Kinematics.from_s(..., per_nucleon=True)        # s = e-N invariant
#     Kinematics.from_y(x, Q2, y) / Kinematics.from_epsilon(x, Q2, eps)

# deuteron polarized along (opposite to) the beam: vector Pz, tensor Pzz
pol = Polarization.beam(kin, "parallel", Pz=0.8, Pzz=0.6)
# or general invariant parameters: Polarization(SL=..., ST=..., phi_S=..., TLL=..., ...)

alpha_p, pT, phi_p = 1.05, 0.15, 0.3        # spectator LF fraction, |p_pT| [GeV], azimuth [rad]
sred = model.reduced_cross_section(kin, alpha_p, pT, phi_p, lambda_e=+0.5, pol=pol)  # Eq. (2.21b)
dsig = model.dsigma(kin, alpha_p, pT, phi_p, lambda_e=+0.5, pol=pol)  # nb/GeV^4 per dx dQ2 dα/α d²pT

# asymmetries built from pure deuteron states (Part I Sec. V D)
model.vector_asymmetry(kin, alpha_p, pT, "parallel")     # A^V_par   -> Eq. (6.4)
model.tensor_asymmetry(kin, alpha_p, pT, "perp")         # A^T_perp  -> Eq. (6.16)

# building blocks
sf = model.structure_functions(kin.x, kin.Q2, alpha_p, pT)   # all IA structure functions (dict)
P = model.lf(alpha_p, pT)                                     # LF neutron distributions, Sec. IV
P.UU, P.TLL_U, P.SL_SL, P.ST_ST_par, P.R_LL                   # Eqs. (4.18)-(4.49)
```

Every argument broadcasts as a numpy array. If you sample the spectator in the deuteron
rest frame, `taggedpd.spectator_lf_from_rest_frame(p_vec)` gives (α_p, p_pT) (Eq. 2.13).

### Proton or neutron tagging

The `tagged` argument selects which nucleon is detected; the other one is the active
nucleon whose structure functions enter:

```python
p_tag = TaggedDIS("AV18", tagged="p")   # default: proton detected, neutron structure (F2n, g1n, ...)
n_tag = TaggedDIS("AV18", tagged="n")   # neutron detected, proton structure (F2p, g1p, ...)
```

Only the nucleon structure functions change. The deuteron side (LF distributions,
kinematics) is identical for both, because the wave functions are isospin symmetric and
the average nucleon mass m is used throughout; `alpha_p`, `pT`, `phi_p` and the
distributions `P.*` then refer to the detected spectator and the active nucleon,
whichever is which. The paper writes all formulas for proton tagging (Eq. 2.2).

## Layout

| module | content | paper |
|---|---|---|
| `wavefunctions.py` | AV18 / CD-Bonn / Paris Yukawa parametrizations, U, W, f₀, f₂ | Eqs. (3.23)–(3.25) |
| `densities.py` | LF neutron distributions P[deuteron, neutron], probabilities R | Sec. IV, Eqs. (4.18)–(4.53) |
| `nucleon.py` | nucleon SF providers: SLAC/Bodek F₂ + R, GRSV2000 g₁ + WW g₂, LHAPDF, grid wrapper | Sec. VI A |
| `crosssection.py` | tagged SFs, Part I master formula, σ_red, dσ, polarization, asymmetries | Secs. II, V, VI; Part I Secs. IV–V |
| `sumrules.py` | integrals over the spectator momentum | Eqs. (4.23)–(4.25), App. A |
| `christy_bosted.py` | port of the Christy–Bosted `F1F209.f` fits: p, n, d F1/F2/R, σ_T, σ_L | PRC 81, 055213; PRC 77, 065206 |
| `photoabsorption.py` | σ_T / F1 of p, n, d: Christy–Bosted plus pluggable models (low W, Regge, …) | — |

## Conventions (read this)

* **Units:** GeV throughout (the C++ code uses MeV).
* **D-wave sign:** the paper's formulas assume a **positive** momentum-space D wave,
  W(k) > 0 at small k. `DeuteronWavefunction` enforces this explicitly, independently of
  the sign convention of the tabulated coefficients (`d_wave_sign`). For the three
  shipped sets it coincides with the C++ `GetWp`.
* **Normalization:** ∫dk k²(U²+W²) = 1, f_L = √(E/4π) U_L, so Eq. (3.23) holds. Yukawa
  parametrizations allow an exact closed form, Σ CᵢCⱼ/(mᵢ+mⱼ); it is evaluated in exact
  rational arithmetic and compared with quadrature in the tests. The published sets are
  normalized to ≲2×10⁻⁴ (AV18 1.00016, CD-Bonn 0.99999983, Paris 1.00013) and are used
  as is, without rescaling.
* **Tensor distributions** are those of the normalized spherical tensors (Eq. 4.19). The
  C++ `getLFdistributions_new` output differs by √(3/2), √2, 1/√2 (T_LL, T_LT, T_TT).
* **Tagging:** `TaggedDIS(..., tagged="p")` (default) or `"n"`; see above. Function and
  attribute names follow the paper's proton-tagging notation in both cases.
* **x** is the rescaled variable x = 2x_D ∈ (0, 2). x_n = x/(2−α_p). The kinematic limit
  α_p < 2 − x gives zero outside it, and unphysical y gives zero cross section.
* **λₑ = ±½** enters as 2λₑ (Part I). Polarization angles are measured relative to the
  lepton plane. `Polarization.beam(...)` implements Part I Sec. V A, with exact γ unless
  `dis_limit=True`.
* **Structure functions:** the paper's IA expressions (5.12), (5.14)–(5.16), (5.26)–(5.27)
  are used verbatim, including the power-suppressed ones (the paper calls these rough
  estimates). T-odd structures are zero in the IA.

## Nucleon structure input

`default_structure()` is SLAC/Bodek F₂ with constant R = 0.18 and the (1+γ²) factor in F₁,
as in the C++ code. Polarized input is GRSV2000 LO g₁ (grid shipped in `taggedpd/data`),
with g₁+g₂ from Wandzura–Wilczek tabulated once at start-up (~0.1 s).

To use another parametrization, subclass `NucleonStructure` and implement `F1F2`, `g1` and
`g1_plus_g2`. `CombinedStructure(unpol, pol)` mixes providers, and `GriddedStructure(p)`
tabulates any provider for speed. `LHAPDFStructure("CT18LO", "NNPDFpol11_100")` gives LO
parton-model F₂ and g₁ from LHAPDF6 sets. It is untested here because the `lhapdf` module
was not installed.

Note: the paper's text quotes DSSV09 for g₁ⁿ, but the C++ code (and thus the figures)
uses GRSV2000 LO. These are different fits (GRSV2000: PRD 63, 094005 (2001); DSSV09:
PRD 80, 034030 (2009)).

## Photoproduction and the resonance region (Christy–Bosted)

```python
from taggedpd import photoproduction
sig = photoproduction(W, "n")                         # sigma_T(gamma n) [microbarn], Q2 = 0
F1d = photoproduction(W, "d", quantity="F1", Q2=0.5)  # deuteron, per nucleus
sig, src = photoproduction(W, "p", return_source=True)  # which parametrization was used per point
```

`taggedpd.christy_bosted` is a vectorized port of `F1F209.f`, the Fortran fits by
P. E. Bosted and M. E. Christy. **Please cite the original work:**

* M. E. Christy and P. E. Bosted, Phys. Rev. C **81**, 055213 (2010), arXiv:0712.3731:
  the proton fit, with σ_T and σ_L separately, 0 ≤ Q² < 8 GeV², 1.1 < W < 3.1 GeV, real
  photons included;
* P. E. Bosted and M. E. Christy, Phys. Rev. C **77**, 065206 (2008), arXiv:0711.0159:
  the deuteron/neutron fit, transverse only, 0 ≤ Q² < 10 GeV², 1.2 < W < 3 GeV.

**How the neutron is handled.** There is no neutron fit. The deuteron fit parametrizes the
free-nucleon average F̄₁ = (F₁ₚ + F₁ₙ)/2. After Fermi smearing with the deuteron momentum
distribution and adding an empirical MEC/dip term, F̄₁ describes the inclusive deuteron
data. The neutron is then F₁ₙ = 2F̄₁ − F₁ₚ(Christy). It takes R = σ_L/σ_T from the proton,
because the deuteron fit is transverse only. The deuteron result is per nucleus and
inelastic only: photodisintegration γd → pn is not included. Details and caveats are in
the module docstring and in `notebooks/photoproduction.ipynb`.

`photoproduction()` evaluates an ordered list of models (`photoabsorption.Piecewise`).
Each point uses the first model whose domain contains it. The default list is:
Christy–Bosted inside its published ranges; the Christy–Bosted forms extrapolated between
the pion threshold and the lower fit edge; and zero below threshold for p and n. Outside
all of them the result is NaN, unless `outside="zero"` or `outside="raise"` is set. To add
a parametrization, subclass `PhotoabsorptionModel` and implement `domain` and `sigma_T`
(or `F1`), then put the model *before* the Christy–Bosted entries. `Blend` gives smooth
transitions between models. See the module docstring for an example.

For tagged DIS in the resonance region, `ChristyBostedStructure()` is a `NucleonStructure`
provider of F1 and F2 in (x, Q²); combine it with a polarized provider through
`CombinedStructure`.

Implementation notes:

* The port reproduces the Fortran compiled with double-precision literals to ~10⁻¹⁵. It
  matches the Fortran as shipped (single-precision literals) to 10⁻⁵ for p and n and
  2×10⁻³ for d.
* `RESMODD` tabulates its resonance shapes in 1 MeV bins of W, which turns n and d into a
  staircase: steps of ~1% above W = 1.2 GeV and tens of percent within 25 MeV of
  threshold. The low-level functions keep this behaviour (`binned=True`) for exact
  regression. `photoproduction` and `ChristyBostedStructure` evaluate the shapes smoothly
  by default.

## Validation (see `tests/` and `notebooks/paper_figures.ipynb`)

* LF distributions vs. C++ `getLFdistributions_new`, all 9 functions on the full
  (α_p, p_T) grid for AV18 / CD-Bonn / Paris: median relative difference ~10⁻⁵. The
  C++ interpolates the wave function on a 1 MeV grid and prints 6 significant digits.
* F₂, g₁ vs. C++ `NucleonStructure`: ~10⁻⁶. WW g₁+g₂ vs. adaptive quadrature and vs.
  the fixed C++: ~10⁻⁴.
* Sum rules (4.23)–(4.25) and Table III integrals: I_[SL] = 0.911, δ∥ = 0.125,
  δ⊥ = 0.0513, ε_[SL] = −0.0571 (paper −0.0568), ε_[ST] = 0.0550.
* x-integrated momentum (5.19), spin (5.29) and Burkhardt–Cottingham (5.34) sum rules
  for the tagged structure functions. These use analytic toy nucleon SFs, so the
  expected values are exact.
* Closed-form asymmetries (6.4), (6.8), (6.12), (6.16) and the Table II extremes reproduced
  by the generic pure-state construction of Part I.
* Christy–Bosted: all 14 outputs (proton F1, R, σ_T, σ_L; nucleon average; F1, F2, R of
  p, n, d) agree with the Fortran on a (W, Q²) grid including Q² = 0, to machine
  precision (`tests/test_christy_bosted.py`, reference tables and generator in
  `tests/reference/`).
* Proton vs neutron tagging: identical deuteron distributions, and every tagged structure
  function scales exactly with the corresponding proton/neutron input (e.g. the
  unpolarized cross-section ratio equals (F_T + εF_L)_p/(F_T + εF_L)_n; A^V_∥ changes
  sign with g₁).

## Issues found in existing material while porting

Items 1 and 2 are fixed in the plot scripts, the C++ and the data (see
[docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md#status-of-fixes)); the paper's figures
still need to be regenerated from them.

1. **Fig. 12 was ~30% too small** in the draft. The plot scripts `tagdeut_ALL_ALT.py`,
   `tagdeut_ALL_ALT_pt.py` and `tagdeut_ALL_g2.py` used `gamma = 2.*x_n*0.93/Q2` instead
   of `/np.sqrt(Q2)`. On top of that, the C++ WW integral `getG1plusg2_grsv2000` was a
   101-point left Riemann sum that overestimated |g₁+g₂| by 7.5% at x = 0.1, and by a
   factor ~3 at x = 0.002 (fixed in physics-code commit `e9d1e6a`). Correct value at
   α_p = 1, p_T = 0: 0.0098 (draft: ≈0.0075). `GRSV2000Structure(ww="cpp")` reproduces
   the old C++ numbers.
2. **Figs. 9 and 15 were 10× too large.** `tagdeut_unpolcross_log.py` and
   `tagdeut_unpolcross_ndep.py` converted to nb with `0.197*0.197*10E07`; `10E07` is 10⁸.
   The model ratio in Fig. 15 is unaffected. Also, "s = 800 GeV²" there is the
   **electron–nucleon** invariant (5×41 GeV), which is what gives ε = 0.98; use
   `Kinematics.from_s(..., per_nucleon=True)`.
3. **Eq. (4.25), "(same for T_LT, T_TT)":** ∫P[T_TT,U] = −0.22 and ∫(2−α_p)P[T_LT,U] = 0.021
   are not zero by themselves. They drop out only through the cos(2φ_p−2φ_TT) and
   cos(φ_p−φ_TL) factors of Eq. (4.22) under the φ_p integration. The "traceless"
   argument applies to T_LL only.
4. **Eq. (5.16c)** writes the partner structure as cos(3φ_p − φ_TT). The master formula
   (Part I Eq. 4.39) has cos(3φ_p − 2φ_TT), and that is what is used here.
5. Minor: ω₂(AV18) = 0.0578 from the parametrization (exact closed form) vs 0.0576 in
   Table III. A 1.2 GeV cutoff (the C++ interpolation range) only gives 0.0577, so the
   origin of the difference is unclear.
