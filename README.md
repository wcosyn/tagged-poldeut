# taggedpd — polarized deuteron DIS with spectator tagging

A standalone, numpy-only Python implementation of the impulse-approximation cross section for

  e(λₑ) + D(S_D, T_D) → e′ + X + p (or n)

from W. Cosyn & C. Weiss, *Semi-inclusive deep-inelastic scattering on a polarized spin-1
target. II. Deuteron and spectator nucleon tagging* (JLAB-THY-26-4661, "the paper"), with
the cross-section decomposition of Part I (same authors). It is a port of the relevant
parts (`Poldeut`, `TDeuteron`/`TYukawaPWF`, `NucleonStructure`) of the C++ code
[`nucqcd`](https://github.com/wcosyn/nucqcd) (formerly `physics-code`), aimed
at use in an MC event generator. There is no numerical integration at run time: about
3 µs per event, vectorized.

Development history, validation details and open items: [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md).

## Install / test

```bash
python3 -m venv --system-site-packages .venv   # system-site: sees the lhapdf module, if installed
source .venv/bin/activate
pip install -e ".[test]"    # runtime: numpy only; test adds pytest + scipy
python -m pytest            # LHAPDF tests skip without lhapdf or the CT18LO / NNPDFpol11_100 sets
python notebooks/build_notebook.py --run   # regenerate notebook + PNGs in notebooks/figures/
python notebooks/build_inclusive_notebook.py --run   # inclusive SFs / photoabsorption / Christy-Bosted notebook
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
| `inclusive.py` | unpolarized inclusive SFs and σ_T, σ_L of p, n, d in (W, Q²), from Q² = 0 to DIS: pluggable models (Christy–Bosted, SLAC/LHAPDF via adapters, low W, Regge, …), adapter to `NucleonStructure` for tagged DIS | — |

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
parton-model F₂ and g₁ from LHAPDF6 sets (tested with those two; needs LHAPDF's python
bindings and `lhapdf install CT18LO NNPDFpol11_100`). Construction takes ~1 s (WW table);
evaluation calls LHAPDF each time, so wrap it in `GriddedStructure` for event generation.

Note: the paper's text quotes DSSV09 for g₁ⁿ, but the C++ code (and thus the figures)
uses GRSV2000 LO. These are different fits (GRSV2000: PRD 63, 094005 (2001); DSSV09:
PRD 80, 034030 (2009)).

## Inclusive structure functions, photoabsorption and the resonance region

```python
from taggedpd import inclusive_sf                     # alias: photoproduction
sig = inclusive_sf(W, "n")                            # sigma_T(gamma n) [microbarn], Q2 = 0
F1d = inclusive_sf(W, "d", quantity="F1", Q2=0.5)     # deuteron, per nucleus
sig, src = inclusive_sf(W, "p", return_source=True)   # which parametrization was used per point
FT, FL = (inclusive_sf(W, "p", q, Q2=1.0) for q in ("F_T", "F_L"))   # paper conventions
```

**Architecture.** All unpolarized inclusive input lives in one layer, `taggedpd.inclusive`,
in the variables (W, Q²). That choice covers real photons as well: in (x, Q²) the point
Q² = 0 is degenerate. Models implement `domain` and `F1` (or `sigma_T`), and `F1F2` when F2
is known. Everything else (σ_L, R, F_T, F_L, …) is derived in one place,
`InclusiveModel.evaluate`. `Piecewise` combines models (first matching domain wins) and
`Blend` makes smooth transitions. Two adapters connect this layer to the (x, Q²)
`NucleonStructure` providers used by `TaggedDIS`:

* `FromNucleonStructure(provider, W_min=…, Q2_min=…)` wraps any provider (SLAC/Bodek,
  LHAPDF, …) as a (W, Q²) model with a kinematic domain;
* `AsNucleonStructure(model)` exposes any model, including a full `Piecewise`, as a
  `NucleonStructure` for `TaggedDIS`. Polarized input is added with `CombinedStructure`.

```python
from taggedpd import (resonance_dis_model, AsNucleonStructure, CombinedStructure,
                      GRSV2000Structure, TaggedDIS)
m = resonance_dis_model()                  # CB in resonance region, blend 2.5-3 GeV, SLAC (R=0.18) above
inclusive_sf(W, "p", "F2", Q2=3.0, model=m)
td = TaggedDIS("AV18", structure=CombinedStructure(AsNucleonStructure(m), GRSV2000Structure()))
# resonance_dis_model(dis=LHAPDFStructure("CT18LO")) for another DIS input
```

The defaults are unchanged: `TaggedDIS` uses SLAC (R = 0.18) + GRSV2000, and
`inclusive_sf` uses the Christy–Bosted 2009 pieces. The combined model gives the same tagged
cross section in DIS kinematics. Where the active nucleon's W is in the resonance region it
differs by tens of percent, up to a factor ~2 at the Δ (notebook, section 8), because the
Bodek fit's resonances are much weaker and broader than Christy–Bosted's.
`ChristyBostedStructure(version=…)` is `AsNucleonStructure` of a single Christy–Bosted fit
(not restricted to its domain).

**Which structure function is returned.** Papers disagree on what "F_T" and "F_L" mean
(factors of x and 2), so every option of `quantity=` is defined explicitly below. All of
them are computed from the fit's own F1 and F2, with the nucleon mass the fit uses in
ν = (W² − M² + Q²)/2M, x = Q²/2Mν and γ² = 4M²x²/Q² = Q²/ν².

| `quantity` | definition | notes |
|---|---|---|
| `"sigma_T"` (default) | σ_T [µb] | Hand flux |
| `"sigma_L"` | R σ_T [µb] | |
| `"R"` | σ_L/σ_T = F_L/F_T | |
| `"F1"` | σ_T (W² − M²)/(8π²α) | the fits' native output |
| `"F2"` | F2 | |
| `"F_T"` | F_[UU,T] = 2F1 | paper / Part I convention |
| `"F_L"` | F_[UU,L] = (1+γ²)F2/x − 2F1 = R F_T | exact relation, including target mass |
| `"F_L_scaling"` | F2/x − 2F1 = F_T (R − γ²)/(1+γ²) | scaling-limit relation (γ → 0), paper Eq. (5.10) |
| `"xF_L"` | (1+γ²)F2 − 2xF1 = x F_[UU,L] | the "F_L" of Christy–Bosted/SIMC and many DIS papers |

In the convention F_T = 2xF1 found elsewhere, with F_L = `"xF_L"` so that
(1+γ²)F2 = F_T + F_L, both functions are x times the paper's. In the resonance region
γ² is not small (≈ 0.5 at Q² = 1 GeV², W = 1.5 GeV), so `"F_L_scaling"` differs from
`"F_L"` and can even be negative: −0.43 vs +0.056 at the Δ for Q² = 1 GeV². Use
`"F_L"` for the physical longitudinal structure function. At Q² = 0, F2, R, σ_L and all
F_L variants are exactly 0. The deuteron values are per nucleus, with x per nucleon.
A model that provides only σ_T (for example a Regge form you add) can return F2- and
F_L-type quantities only at Q² = 0; at Q² > 0 it raises `NotImplementedError`.

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
the module docstring and in `notebooks/inclusive.ipynb`.

The default model of `inclusive_sf()` (`default_photoabsorption()`) is an ordered
`Piecewise` list. Each point uses the first model whose domain contains it. The list is:
Christy–Bosted inside its published ranges; the Christy–Bosted forms extrapolated between
the pion threshold and the lower fit edge; and zero below threshold for p and n. Outside
all of them the result is NaN, unless `outside="zero"` or `outside="raise"` is set. To add
a parametrization, subclass `InclusiveModel` and implement `domain` and `sigma_T`
(or `F1`), plus `F1F2` if F2 is known. Then put the model *before* the Christy–Bosted
entries. `Blend` gives smooth transitions between models. See the module docstring for an
example.

Implementation notes:

* The port reproduces the Fortran compiled with double-precision literals to ~10⁻¹⁵. It
  matches the Fortran as shipped (single-precision literals) to 10⁻⁵ for p and n and
  2×10⁻³ for d.
* `RESMODD` tabulates its resonance shapes in 1 MeV bins of W, which turns n and d into a
  staircase: steps of ~1% above W = 1.2 GeV and tens of percent within 25 MeV of
  threshold. The low-level functions keep this behaviour (`binned=True`) for exact
  regression. `inclusive_sf` and `ChristyBostedStructure` evaluate the shapes smoothly
  by default.
* `AsNucleonStructure` (and therefore `ChristyBostedStructure`) converts x → W with the
  physical proton or neutron mass, as nucqcd's `NucleonStructure::getF_CB` does.
  Until 2026-11-19 `ChristyBostedStructure` used the fit's proton mass for the neutron as
  well; the switch changed the neutron by up to 1–2% near threshold (large x).

### PRELIMINARY: the 2021 fit (F1F221)

> **Preliminary and unpublished.** Do not use this fit in published work without checking
> its status with the author, M. E. Christy. The 2009 fits above remain the default and
> the citable reference.

`taggedpd.christy_bosted21` ports `F1F2IN21_v1.0.f`, "F1F221 v1.0 prerelease 1"
(M. E. Christy, April 2022), as distributed with the JLab Hall C simulation package SIMC:
https://github.com/JeffersonLab/simc_gfortran. According to its header, the
deuteron/neutron fit is by M. E. Christy, N. Kalantarians, J. Ethier and W. Melnitchouk
("to be published"). It is based on inclusive deuteron data and BONuS tagged n/d data,
and includes photoproduction at Q² = 0.

Differences from the 2009 fits:
* the neutron has its own fit, with its own σ_T and σ_L;
* the deuteron is p + n smeared with CD-Bonn light-cone distributions, with no MEC term;
* the range extends to W² < 20 and Q² < 30 GeV².

The smearing tables in `taggedpd/data/f1f2tables/` come from the same SIMC repository.
That repository has no license file, so redistribution has not been cleared.

```python
from taggedpd import default_photoabsorption, inclusive_sf, ChristyBostedStructure, resonance_dis_model
m21 = default_photoabsorption(fit="2021")         # emits PreliminaryFitWarning
inclusive_sf(W, "n", model=m21)
ChristyBostedStructure(version="2021")            # resonance-region F1, F2 (warns too)
resonance_dis_model(fit="2021")                   # 2021 fit in the resonance region (warns too)
```

The port agrees with the Fortran to rounding. Quirks of the original are reproduced on
purpose and documented in the module:
* F2 is 0/0 at Q² = 0 in the Fortran; the port returns 0;
* `GETFY` reads the smearing table one y bin too high;
* the deuteron F1 jumps at Q² = 0.05 GeV² (separate low-Q² smearing routine) by ~1% above
  the Δ and more near threshold;
* the deuteron F1 has 1–5% numerical steps as a function of Q² near threshold.

The comparison with the 2009 fits is in `notebooks/inclusive.ipynb`, section 7.

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
   factor ~3 at x = 0.002 (fixed in nucqcd commit `e9d1e6a`). Correct value at
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
