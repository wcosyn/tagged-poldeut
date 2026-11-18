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
| WW g1+g2 | accurate tabulated integral by default; `ww="cpp"` reproduces the C++ Riemann sum | the C++ sum is inaccurate (see issues below) |
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
  factor. The WW table agrees with adaptive quadrature to ~10⁻⁴.
- **Cross section.** The x-integrated momentum (5.19), spin (5.29) and Burkhardt–Cottingham
  (5.34) sum rules hold. These are tested with analytic toy nucleon structure functions,
  so the expected values are exact. The closed-form asymmetries (6.4), (6.8), (6.12) and
  (6.16) and the Table II extremes are reproduced by the generic pure-state construction
  of Part I. The φ_p average of the master formula equals the analytic ⟨F⟩₁. dσ equals
  the C++ `calc_dsigma_unpol` expression exactly.
- **Notebook.** It executes cleanly in Jupyter (kernel "Python (taggedpd)", local
  `.venv`). Figs. 3–8, 10, 11, 13, 14, 16 and 17 agree with the paper; Figs. 9, 12 and 15
  differ for the reasons below.
- **Speed.** About 3 µs per event for the full σ_red, vectorized; start-up ~0.1 s.

## Issues found in existing material

These affect the paper draft and should be fixed before publication.

1. **Fig. 12 is ~30% too small.** `tagdeut_ALL_ALT.py`, `tagdeut_ALL_ALT_pt.py` and
   `tagdeut_ALL_g2.py` use `gamma = 2.*x_n*0.93/Q2`, which should be `/np.sqrt(Q2)`. On
   top of that, `NucleonStructure::getG1plusg2_grsv2000` (a 101-point left Riemann sum)
   overestimates |g₁+g₂| by 7.5% at x = 0.1 and by ~3× at x = 0.002. At α_p = 1,
   p_T = 0 the correct value is 0.0098; the figure shows ≈0.0075. The g2 figure from
   `tagdeut_ALL_g2.py` is affected in the same way.
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

## Known limitations and open items

- Impulse approximation only: no FSI, so the T-odd structures are zero. The
  power-suppressed structure functions are rough estimates, as the paper states.
- `LHAPDFStructure` is untested because the `lhapdf` module was not available. For event
  generation wrap it in `GriddedStructure`.
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
