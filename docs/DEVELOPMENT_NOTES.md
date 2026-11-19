# Development notes

Record of how `taggedpd` was built, which choices were made and why, how it was validated,
and what was found along the way. For usage and conventions see the [README](../README.md).

## Origin and scope

`taggedpd` is a standalone Python port of the parts of the C++ `physics-code` repository
needed for tagged DIS on the polarized deuteron, written to serve as the physics engine
of an MC generator. The requirements were:

- reproduce the (reduced) cross section of Cosyn & Weiss, *SIDIS on a polarized spin-1
  target. II* (JLAB-THY-26-4661, "the paper"), using the Part I master formula;
- use the deuteron LF densities of `Poldeut::getLFdistributions_new` with the AV18,
  CD-Bonn and Paris Yukawa parametrizations, requiring a **positive D wave**;
- test the wave-function normalization, Eq. (3.23), against an exact result;
- use the nucleon F1, F2, g1, g2 of `NucleonStructure`;
- be fast and lightweight, with no integrations at run time (the WW g2 is gridded);
- provide a notebook reproducing the paper's figures as a cross-check;
- comment the code amply.

### Decisions taken

| Question | Decision | Rationale |
|---|---|---|
| Repository | `tagged-poldeut`, package `taggedpd`, private on GitHub | — |
| g1 input | GRSV2000 LO (the grid already in `physics-code/share/grsv`), behind a pluggable interface, plus an LHAPDF backend | It is what the C++, and therefore the figures, actually use. The paper text says DSSV09, a different fit. |
| R = σ_L/σ_T | constant R = 0.18 by default; E143 `r1998` available | matches Sec. VI A of the paper |
| F1 target-mass factor | (1+γ²) included by default (`target_mass=True`) | the C++ code that produced the figures includes it; `False` gives the strict DIS limit |
| WW g1+g2 | accurate tabulated integral by default; `ww="cpp"` reproduces the old C++ Riemann sum | the old C++ sum was inaccurate (issue 1 below, now fixed in physics-code) |
| Master formula | taken from Part I Eqs. (4.36)–(4.39) | the paper only quotes it |
| Wave functions | used as published, not rescaled to norm 1 | the norms are 1 to within 2×10⁻⁴; this keeps agreement with the C++ |

## Mapping from `physics-code`

| `physics-code` | `taggedpd` | Notes |
|---|---|---|
| `Tools/TDeuteron.cpp`, `TYukawaPWF.cpp` | `wavefunctions.py` | Boundary conditions are solved as in the C++ constructor. W is evaluated analytically; the C++ interpolates on a 0–1200 MeV, 1 MeV grid. |
| `DIS/Poldeut.cpp::getLFdistributions_new` | `densities.py` | Paper equations coded directly. The C++ C0L/D0T forms are numerically identical. The C++ tensor outputs carry extra factors √(3/2), √2, 1/√2, and its internal names `P_STST_par`/`perp` are swapped relative to the header (callers get the right values). |
| `Tools/NucleonStructure.cpp` (`f2n_b`, `f2p_b`, `bodek`, `getF_SLAC`) | `nucleon.SLACStructure` | The goto logic of `bodek` is rewritten with numpy masks. |
| `Tools/grsv2000pdf_g1.f` (`PARPOL`, `DFINT`) | `nucleon.GRSV2000Structure` | Same bilinear interpolation in (ln x, ln Q²); out-of-range x, Q² are clamped rather than stopping. |
| `Tools/R199x.f` (`R1998`) | `nucleon.r1998` | — |
| plot scripts in `~/Calculations/poldeut/plots` | `notebooks/build_notebook.py` | — |

## Validation

Reference data in `tests/reference/` is subsampled C++ output: `lf_dens_24_*_fine*.txt` and
`neutron_SF.txt` from `~/Calculations/poldeut/results`.

- **Wave functions.** Eq. (3.23) by quadrature agrees with the closed form
  Σ CᵢCⱼ/(mᵢ+mⱼ) to 10⁻⁷. The closed form is evaluated in exact rational arithmetic,
  because for Paris float64 loses ~10⁻⁶ to cancellation between terms of order 10¹⁰;
  it was cross-checked against 40-digit mpmath. Norms: AV18 1.00016, CD-Bonn 0.9999998,
  Paris 1.00013. The D wave is positive for all three sets (`d_wave_sign = +1`, same as the C++).
- **LF densities vs C++.** All 9 distributions on the full (α_p, p_T) grid: median relative
  difference ~10⁻⁵. Larger deviations, up to ~2%, appear only near nodes, where values cross zero.
- **Sum rules / Table III (AV18).** I_[SL] = 0.9108 (paper 0.911), δ∥ = 0.1252 (0.125),
  δ⊥ = 0.0513 (0.0512), ε_[SL] = −0.0571 (−0.0568), ε_[ST] = 0.0550 (0.0550). Eqs. (4.23)
  and (4.24) equal the wave-function norm exactly. Note that AV18 has a slow high-k tail,
  with 4×10⁻⁵ of the norm above 3 GeV.
- **Nucleon SFs vs C++.** F2 and g1 agree to ~10⁻⁶. F1 agrees to 10⁻⁴ with the target-mass
  factor. The WW table agrees with adaptive quadrature to ~10⁻⁴, and with the fixed C++
  (`tests/reference/neutron_SF.txt.gz`) to ~10⁻⁴ (up to ~10⁻⁵ absolute at the sign change
  of g1+g2 near x ≈ 0.35, where the C++ is the more accurate of the two).
  `tests/reference/neutron_SF_oldWW.txt.gz` keeps the pre-fix C++ output, reproduced by
  `ww="cpp"`.
- **Cross section.** The x-integrated momentum (5.19), spin (5.29) and Burkhardt–Cottingham
  (5.34) sum rules hold. These are tested with analytic toy nucleon structure functions,
  so the expected values are exact. The closed-form asymmetries (6.4), (6.8), (6.12) and
  (6.16) and the Table II extremes are reproduced by the generic pure-state construction
  of Part I. The φ_p average of the master formula equals the analytic ⟨F⟩₁. dσ equals
  the C++ `calc_dsigma_unpol` expression exactly. Neutron tagging (`tagged="n"`) is
  tested against proton tagging: same LF distributions, and each structure function
  scales with the ratio of the proton/neutron input it contains.
- **Notebook.** It executes cleanly in Jupyter (kernel "Python (taggedpd)", local
  `.venv`). Figs. 3–8, 10, 11, 13, 14, 16 and 17 agree with the paper; Figs. 9, 12 and 15
  differ for the reasons below.
- **Speed.** About 3 µs per event for the full σ_red, vectorized; start-up ~0.1 s.

## Issues found in existing material

These affect the paper draft and should be fixed before publication.

1. **Fig. 12 is ~30% too small.** `tagdeut_ALL_ALT.py`, `tagdeut_ALL_ALT_pt.py` and
   `tagdeut_ALL_g2.py` used `gamma = 2.*x_n*0.93/Q2`, which should be `/np.sqrt(Q2)`. On
   top of that, `NucleonStructure::getG1plusg2_grsv2000` (a 101-point left Riemann sum)
   overestimated |g₁+g₂| by 7.5% at x = 0.1 and by ~3× at x = 0.002. At α_p = 1,
   p_T = 0 the correct value is 0.0098; the figure shows ≈0.0075. The g2 figure from
   `tagdeut_ALL_g2.py` is affected in the same way. *(Fixed, see below.)*
2. **Figs. 9 and 15 are 10× too large.** `tagdeut_unpolcross_log.py` and
   `tagdeut_unpolcross_ndep.py` convert to nb with `0.197*0.197*10E07`; `10E07` is 10⁸,
   where the conversion needs 10⁷. The same factor is in `tagdeut_obs.py` and
   `tagdeut_obs_v2.py`. The CD-Bonn/AV18 ratio of Fig. 15 is unaffected.
3. **Fig. 9 caption.** "s = 800 GeV²" is the electron–nucleon invariant (5×41 GeV, script
   `y = Q2/(x*s)`); this is what gives ε = 0.98. `Kinematics.from_s(..., per_nucleon=True)`
   implements this convention exactly.
4. **g1 parametrization.** The text (Sec. VI A) cites DSSV09; the code uses GRSV2000 LO.
5. **Eq. (4.25), "(same for T_LT, T_TT)".** ∫P[T_TT,U] = −0.22 and ∫(2−α_p)P[T_LT,U] = 0.021.
   Both vanish only together with the azimuthal factors of Eq. (4.22) under φ_p
   integration. The "traceless" argument applies to T_LL only.
6. **Eq. (5.16c)** pairs cos(φ_p − 2φ_TT) with cos(3φ_p − φ_TT). Part I's master formula
   has cos(3φ_p − 2φ_TT), which is what is used here.
7. **Table III, ω₂(AV18) = 0.0576.** The parametrization gives 0.0578 exactly; a 1.2 GeV
   cutoff gives 0.0577. The origin of the difference is unclear.

## Status of fixes

State on 2026-11-18. Issues 1 and 2 are fixed at the source; the figures themselves have
not been regenerated yet.

- **Plot scripts** (`~/Calculations/poldeut/plots`, not under version control; originals in
  `plots/backup_2026-11-18_before_fixes/`):
  - γ_n → `2*x_n*m/np.sqrt(Q2)` in `tagdeut_ALL_ALT.py`, `tagdeut_ALL_ALT_pt.py`,
    `tagdeut_ALL_g2.py` and `tagdeut_ALL_nucldep.py`;
  - `tagdeut_ALT_study.py`: the same √Q² fix in γ and γ_n, and γ → γ_n in the `F_LST`
    lines. That last change follows Eq. (5.27), where the neutron structure function
    carries γ_n; it is a judgement call, so check it against the intent of that study;
  - `10E07` → `1E07` in `tagdeut_unpolcross_log.py`, `tagdeut_unpolcross_ndep.py`,
    `tagdeut_obs.py` and `tagdeut_obs_v2.py`.
- **C++** (physics-code, branch `fix-ww-integral`, commit `e9d1e6a`, not yet merged or
  pushed): Gauss–Legendre WW integral in both `NucleonStructure` and the older
  `NuclStructure` (used by `Poldeut` and `DeuteronStructure`); new table generator
  `progs/poldeut/neutron_sf`.
- **Data**: `~/Calculations/poldeut/results/neutron_SF.txt` and `neutron_SF_more.txt`
  regenerated with `neutron_sf 100` / `neutron_sf 300`. Only the g1+g2 column changed.
  Old files in `results/backup_2026-11-18_old_WW/`.
- **Check**: the fixed `tagdeut_ALL_ALT.py` on the new data agrees with `taggedpd` for
  Fig. 12 to 4 digits (0.00985 at α_p = 1, p_T = 0); the fixed `tagdeut_unpolcross_log.py`
  agrees with `taggedpd` for Fig. 9 to 0.3% (rounded ħc and α in the script).

Still open:

- Regenerate Figs. 9, 12 and 15 (and the g2 study figure) for the paper.
- matplotlib `usetex` needs the `cm-super` fonts, which are missing on the upgraded system
  (`sudo apt install cm-super`); without them some figures fail to save
  (e.g. `ALT_nodepol_v2.pdf`).
- physics-code was reinstalled into `lib/` and `bin/` after the OS upgrade (the old
  install linked GSL 27, which is gone). `GPD_Partons` and `progs/GPD` do not build:
  the installed PARTONS headers need SFML, and `diDVCS_gen_gridonly` lacks the ROOT
  include path. `lib/libGPD.so` and `bin/conv_T_test` are therefore still the old,
  non-loading builds.

## Photoproduction branch (2026-11-19)

On branch `photoproduction`: a Python port of the Christy–Bosted resonance fits and a
photoabsorption interface that later parametrizations can extend. File and function names
below are the current ones, after the refactor described at the end of this section:
`photoabsorption.py` became `inclusive.py`, and `photoproduction()` became `inclusive_sf()`,
with `photoproduction` kept as an alias.

### What was added

| file | content |
|---|---|
| `taggedpd/christy_bosted.py` | Port of `F1F209.f` (P. E. Bosted, M. E. Christy), in physics-code as `Tools/F1F209_test.f`: `RESMOD507`, `CHRISTY507`, `RESMODD`, `MEC`, `RESD`, `PIND`, and the A ≤ 2 branches of `F1F2IN09`. |
| `taggedpd/inclusive.py` | `inclusive_sf(W, target, quantity, Q2)` (alias `photoproduction`), the `InclusiveModel` interface (alias `PhotoabsorptionModel`), `ChristyBosted`, `BelowThreshold`, `Blend`, `Piecewise`, `default_photoabsorption()`. |
| `taggedpd/inclusive.py` (re-exported from `nucleon`) | `ChristyBostedStructure`: p and n F1, F2 in (x, Q²) as a `NucleonStructure` provider. |
| `tests/test_christy_bosted.py` | Regression against the Fortran, the Q² → 0 limit, the neutron construction, and the interface. |
| `tests/reference/christy_bosted_{sp,dp}.txt.gz`, `tests/reference/fortran/` | Fortran reference tables, the driver, and build instructions. |
| `notebooks/build_inclusive_notebook.py` → `inclusive.ipynb` | Plots of σ_T and F1 for p, n, d, the neutron construction, small Q², binning, and extending the model. |

References (cited in the module, README and notebook): Christy & Bosted, PRC 81, 055213
(2010), arXiv:0712.3731 (proton); Bosted & Christy, PRC 77, 065206 (2008),
arXiv:0711.0159 (deuteron and neutron). The fit ranges quoted are taken from these
abstracts.

### Decisions

| Question | Decision | Rationale |
|---|---|---|
| Which Fortran routines | Those needed for p, n, d: everything except the A > 2 branch, `F1F2QE09` and `RESDER` | `RESDER` needs `F1F207D2emat.dat`, which is not available; QE does not describe photodisintegration |
| Single-precision literals in the Fortran | Use double precision in Python | Agreement with the double-literal Fortran is ~10⁻¹⁵; with the Fortran as shipped, ≤10⁻⁵ (p, n) and ≤2×10⁻³ (d, where W bins shift) |
| 1 MeV W binning in `RESMODD` | `binned=True` in the low-level port (exact regression); smooth evaluation in `inclusive_sf` (formerly `photoproduction` and `ChristyBostedStructure` | The binning is a table artifact. It produces a staircase in n and d (~1% steps above W = 1.2, up to tens of % within 25 MeV of threshold) |
| Default validity domain | The published fit ranges. Between the pion threshold and the lower fit edge, the same functions are used, labelled "threshold extrapolation". Zero below threshold for p and n; NaN outside all pieces | The user wants Christy–Bosted where it is valid, with room for low-W and high-W (Regge) models; NaN makes coverage gaps visible |
| Model selection | Ordered `Piecewise` list, first matching domain wins; `return_source=True` reports which piece was used | New parametrizations are inserted without touching existing ones |
| Module name | `photoabsorption.py` at first, now `inclusive.py` (see the refactor below) | Exporting a function with the same name as its module from `__init__` would shadow the submodule |
| `ChristyBostedStructure.FT_FL` | Returns (2F₁, 2F₁R) directly from the fit | Rebuilding F_L as (1+γ²)F₂/x − 2F₁ with the physical neutron mass is off by ~0.5%, because the fit uses the proton mass |

### Validation

- All 14 output columns (proton F1, R, σ_T, σ_L; F̄₁; F1, F2, R of p, n, d) are compared
  with the Fortran on 2938 (W, Q²) points: W from 0.95 to 3.2 GeV, 13 values of Q² from 0
  to 8 GeV². Maximum relative difference: 3×10⁻¹⁵ against the double-literal build,
  1×10⁻⁵ (p, n) and 9×10⁻⁴ (d) against the build as shipped.
- At Q² = 0: σ_L = F2 = R = 0, σ_T > 0, and σ_T ↔ F1 equals the fit's own conversion. The
  γp Δ peak is 486 µb at W = 1.210 GeV.
- The baseline suite (53 passed, 1 skipped) is unchanged; the full suite is 69 passed,
  1 skipped.
- Speed: p ~1.1 µs per point, n ~1.4 µs, d ~27 µs. The deuteron smearing loop was
  restricted to the points in each bin, which made it about 2.6× faster.

### Observations about the fits (not bugs in the port)

1. **Steep Q² dependence near Q² = 0.** Example: the height of the proton's second
   resonance has the factor 1 + 6823 Q²/(1 + 33521 Q²). At W = 1.5 the proton σ_T changes
   by +22% and the neutron by −30% between Q² = 0 and 10⁻³ GeV². For quasi-real photons
   the Q² = 0 value is therefore not the right one.
2. **The deuteron below W ≈ 1.2 at Q² = 0 is dominated by the MEC term**, ~300 µb near
   threshold. It uses max(0.3, Q²), so for real photons it is frozen at Q² = 0.3, and it
   lies outside the deuteron fit's W range. It is not constrained by real-photon data
   there.
3. **σ(γn)/σ(γp) ≈ 1.15 at the Δ peak.** The Δ is isovector, so this most likely comes
   from the unfolding (smearing and MEC) used to extract the neutron.
4. The physics-code C++ (`NucleonStructure::getF_CB`, `NuclStructure`) calls `F1F2IN09`
   in (x, Q²). Its W² = M² + Q²(1/x − 1) equals M² at Q² = 0, below the 1.155 GeV² cut, so
   real photons cannot be reached through that interface. The proton is passed as Z = 1,
   A = 0; this works only because only `int(A) < 2` is tested.

### PRELIMINARY 2021 fit (F1F221), added 2026-11-19

**Status: preliminary and unpublished.** W. Cosyn will check with M. E. Christy before any
use in published work; until then the 2009 fits remain the default and the reference.

- **Search for a newer fit.** No published successor to PRC 81, 055213 / PRC 77, 065206
  was found. The newer version exists only as code in JLab SIMC
  (https://github.com/JeffersonLab/simc_gfortran):
  - `F1F2IN21.f`: "F1F220 v0.995", 2021-02-20, committed 2021-06;
  - `F1F2IN21_v1.0.f`: "F1F221 v1.0 prerelease 1", 2022-04-04, committed 2022-07, with
    proton fit "120521" and neutron fit 2021-11-16. **This is the version ported.**

  Its header attributes the deuteron/neutron fit to Christy, Kalantarians, Ethier and
  Melnitchouk, "to be published"; no such paper was found. F1F221 has been used, for
  example, in the recent JLab measurement of the deuteron-to-proton structure-function
  ratio. The SIMC repository has no license file.
- **What was added.**
  - `taggedpd/christy_bosted21.py`: `RESMODP/N`, `RESCSP/N`, `SF`, `GETFY`, `DSIMPS`,
    `SMEARSUB`, `SMEARSUBLOWQ`, `RESCSD`, and `F1F2IN21` for A ≤ 2.
  - The CD-Bonn on-shell tables, gzipped, in `taggedpd/data/f1f2tables/`.
  - `ChristyBosted2021` in `inclusive.py`, `default_photoabsorption(fit="2021")`,
    and `ChristyBostedStructure(version="2021")`. All emit `PreliminaryFitWarning`, and
    every name carries "(PRELIMINARY)".
  - Tests in `tests/test_christy_bosted21.py`, with reference tables generated by
    `tests/reference/fortran/cb21_reference.f`.
  - Notebook section 7, which compares the 2009 and 2021 fits.
- **Not ported.** The quasi-elastic part (`F1F2QE21`, `SQESUB`), off-shell smearing (switched
  off in `RESCSD` in this version) and A > 2.
- **Validation.** All 10 output columns agree with the double-literal Fortran to ≤ 8×10⁻¹⁶
  on 3616 points, with Q² from 0 to 25 GeV². Against the Fortran as shipped they agree to
  ≤ 2×10⁻⁵, except at exactly Q² = 0.05. There the shipped build compares
  `q2 .LT. 0.05` with float32(0.05) > 0.05 and takes the low-Q² branch; that case is also
  tested. The smearing functions integrate to 1 at γ = 1. The 2021 and 2009 proton fits
  agree within 15% across the resonance region at Q² = 0.
- **Quirks of the original, reproduced on purpose.**
  1. `SF` computes F2 as 0/0 at Q² = 0; the port returns F2 = 0 there (the only deviation).
  2. `GETFY` uses bin index `int(500 y) + 1`, so the "interpolation" is an extrapolation
     from the next pair of y nodes.
  3. Because `SMEARSUBLOWQ` (Q² < 0.05) integrates F1(z) while `SMEARSUB` integrates
     2zF1/(2x) = F1/y, the deuteron F1 jumps at Q² = 0.05 GeV²: by −1% to −0.5% above the Δ,
     +3% to +9% at W = 1.15–1.08, and up to a factor of 2 at W ≈ 1.0.
  4. The deuteron F1 has 1–5% steps as a function of Q² near threshold. They come from the
     fixed 48/60/76-node Simpson grid crossing the 0.002 table bins.
  5. For γ > 5 (outside the tables) the Fortran reads zero-padded common-block entries; the
     port does the same, with γ clamped at the array size.
  6. `if((w-mp).LT.wdif(1)) sig = 0` in `RESMODP/N` can never be true.
  7. `GETFY` only reads the tables if `SQESUB(..., first=.true.)` was called beforehand. If
     `F1F2IN21` is called without that initialization, the deuteron is silently zero.
- **Physics differences from 2009, Q² = 0.** The proton agrees within ~5% except near
  threshold. The neutron differs by ±20%: σn/σp at the Δ is 0.95 (2009: 1.15), and near
  threshold the 2021 neutron is about twice the 2009 one. The 2021 neutron also has a cusp
  at the η threshold, where the exponent t₁ of its non-resonant η term is ≈ 0.03. The 2021
  deuteron has no strength below the free-nucleon threshold (no MEC).
- **Speed.** p and n ~1 µs per point; the deuteron ~110 µs per point.

### Structure-function conventions in `inclusive_sf` (added 2026-11-19)

`quantity=` now also accepts `"sigma_L"`, `"R"`, `"F2"`, `"F_T"`, `"F_L"`, `"F_L_scaling"`
and `"xF_L"`. The definitions are in the README table and in the `inclusive` module
docstring.

| Question | Decision | Rationale |
|---|---|---|
| Normalization of F_T and F_L | The paper's: F_T = 2F1, F_L = (1+γ²)F2/x − 2F1 | Matches `NucleonStructure.FT_FL` and the tagged-DIS code. The x-weighted F_L that Christy–Bosted and SIMC use is available as `"xF_L"` |
| "With or without the scaling limit" | `"F_L"` is the exact relation, equal to R·F_T; `"F_L_scaling"` = F2/x − 2F1 is Eq. (5.10) | In the resonance region γ² = Q²/ν² is not small, and the scaling form can turn negative: −0.43 vs +0.056 at the Δ for Q² = 1 GeV² |
| Mass in x, ν and γ | The mass each fit uses itself (`kin_mass`): 2009, 0.93828 for p, n and d; 2021, 0.938272 for p and n and the average nucleon mass for d | With this choice F_L = R·F_T holds to rounding against each fit's own R, and the deuteron `xF_L` reproduces RESCSD's FL |
| Q² = 0 | F2, R, σ_L and all F_L variants are exactly 0 | These are the limits; x = 0 there |
| Models that provide only σ_T | F_T works everywhere; F2- and F_L-type quantities only at Q² = 0, otherwise `NotImplementedError` | Avoids silently inventing an R for a future Regge model |
| `Blend` | Interpolates each quantity separately | Simple. The blended R is then not exactly σ_L/σ_T of the blended cross sections; this is documented |

Tests: `test_quantity_conventions_2009` (p, n, d at three values of Q²; every identity in
the table to ≤10⁻⁹), `test_quantity_conventions_2021`,
`test_scaling_relation_breaks_down_at_low_Q2`, `test_quantities_at_real_photon_point`,
and two tests for models that provide only σ_T.

### Environment note

On this machine `python3 -m venv` cannot bootstrap pip (`python3.12-venv` is not
installed). The `.venv` was created with `--system-site-packages --without-pip`, and the
system pip was then used to install pytest, numpy 2, scipy and matplotlib into it. With
the system numpy 1.26, `tests/test_nucleon.py::test_ww_integral_identity` fails because
`np.trapezoid` needs numpy ≥ 2, although `pyproject.toml` declares numpy ≥ 1.22.

### One inclusive layer for photoproduction and tagged DIS (refactor, 2026-11-19)

**Motivation (W. Cosyn).** The Christy–Bosted fits should be usable in tagged DIS at finite
Q² as well as for real photons, and the code should stay as general as possible.

**Before the refactor** there were two parallel interfaces sharing the Christy–Bosted code:
`NucleonStructure`, in (x, Q²) for p and n, used by `TaggedDIS`; and `PhotoabsorptionModel`,
in (W, Q²) for p, n and d, with region switching and the conventions table. `TaggedDIS`
could use only one provider everywhere, SLAC/LHAPDF could not be combined with
Christy–Bosted, and F_T/F_L were defined in two places.

**After the refactor:**

| Piece | Role |
|---|---|
| `taggedpd/inclusive.py` (renamed from `photoabsorption.py`) | The single home for unpolarized inclusive input in (W, Q²). (W, Q²) is used because Q² = 0 is degenerate in (x, Q²). `inclusive_sf()` is the entry point; `photoproduction` and `PhotoabsorptionModel` remain as aliases |
| `InclusiveModel.evaluate` | The only place where σ_L, R, F_T, F_L, F_L_scaling and xF_L are derived from F1 and F2 |
| `FromNucleonStructure(provider, W/Q² domain)` | Any (x, Q²) provider (SLAC, LHAPDF, …) as a (W, Q²) piece. Default Q² ≥ 1, since DIS fits do not describe real photons |
| `AsNucleonStructure(model, target_mass=True)` | Any model or `Piecewise` as a `NucleonStructure`; `FT_FL` returns `"F_T"` together with `"F_L"`, or with `"F_L_scaling"` if `target_mass=False` |
| `resonance_dis_model(dis=SLAC, W_blend=(2.5, 3.0), fit="2009")` | `Blend` from CB to DIS over the window (only where both apply), DIS above it, the CB pieces, then DIS as catch-all (e.g. Q² > 8). The deuteron uses the CB pieces only |
| `ChristyBostedStructure` | Now `AsNucleonStructure` of a single CB fit, defined in `inclusive.py` and lazily re-exported from `nucleon` (avoids a circular import) |
| `Blend.domain` | Now the intersection of the W window with *both* models' domains |
| `notebooks/inclusive.ipynb` (renamed) | New section 8: resonance + DIS model and its effect in tagged DIS |

**Checks.**
- About 8000 outputs (all quantities of `inclusive_sf` for p, n, d at three Q² values for
  both fits; `ChristyBostedStructure`; `TaggedDIS` with the default and with the CB
  structure) were saved before the refactor and compared after it. Everything is
  bit-identical (≤ 8×10⁻¹⁵) except the deliberate change below.
- Wrapping SLAC through `FromNucleonStructure` and `AsNucleonStructure` reproduces it to
  10⁻¹⁴, and `TaggedDIS` gives an identical σ_red (`tests/test_inclusive.py`).
- The blend is continuous at both edges.
- In DIS kinematics the combined model reproduces SLAC exactly, and in the resonance region
  it reproduces `ChristyBostedStructure`.

**Deliberate change.** `ChristyBostedStructure` now converts x → W with the physical
neutron mass for the neutron, like physics-code's `NucleonStructure::getF_CB` and the
generic adapter. It previously used the fit's proton mass for both nucleons. The neutron
changes by up to 1–2% near threshold (large x); the proton by < 10⁻⁵.

**Physics observation.** Compared with SLAC everywhere (the paper's input), Christy–Bosted
in the active neutron's resonance region changes the tagged σ_red by tens of percent, and
by up to a factor ~2 at the Δ (x ≈ 0.8, Q² = 2), because the Bodek fit's neutron resonances
are much weaker and broader. In the blend window the two fits agree within ~5% for the
proton and ~13% for the neutron at Q² = 1.

**Open.** No model covers real photons above W ≈ 3 GeV, or Q² < 1 above the CB range; these
return NaN by default. That region is reserved for a Regge-type model.

## Known limitations and open items

- Impulse approximation only: no FSI, so the T-odd structures are zero. The
  power-suppressed structure functions are rough estimates, as the paper states.
- `LHAPDFStructure` is tested (`tests/test_lhapdf.py`) with CT18LO and NNPDFpol11_100 only:
  parton-model sums and isospin against direct LHAPDF calls, the WW table against
  quadrature, gridded vs direct, and rough agreement with SLAC F₂ / GRSV2000 g₁ (within
  ~14% for 0.1 ≤ x ≤ 0.5 at Q² = 10). Testing it fixed two bugs: a scalar x with an array of
  Q² raised, and scalars came back as shape-(1,) arrays. LHAPDF is now called vectorized
  (construction 8 s → 0.7 s). It is LO only (no NLO coefficient functions, no target-mass
  factor), and uses LHAPDF's extrapolation outside a set's range. For event generation wrap
  it in `GriddedStructure`.
- DSSV09 is not implemented. Adding it needs its grids, wrapped as a `NucleonStructure`
  subclass.
- The LF wave function uses the nonrelativistic approximation, Eq. (3.24), which the
  paper considers accurate for |k| ≲ 300 MeV. Outside the kinematic limit α_p < 2 − x,
  and for unphysical y, the cross section is returned as 0.
- `GRSV2000Structure` clamps x < 10⁻⁴ and Q² outside [0.8, 10⁶] GeV² to the grid edges.
  Check `GRSV2000Structure.in_range` if this matters.

## Environment

```bash
cd ~/Git/tagged-poldeut
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" jupyterlab nbconvert ipykernel matplotlib
python -m ipykernel install --user --name taggedpd --display-name "Python (taggedpd)"
python -m pytest
python notebooks/build_notebook.py          # regenerate paper_figures.ipynb from its source
jupyter nbconvert --to notebook --execute --output paper_figures.executed.ipynb notebooks/paper_figures.ipynb
```

`notebooks/paper_figures.ipynb` is generated from `notebooks/build_notebook.py`; edit the
builder, not the notebook. Executed copies (`*.executed.ipynb`) and `notebooks/figures/`
stay local (git-ignored).
