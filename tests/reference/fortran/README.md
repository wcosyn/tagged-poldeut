# Christy–Bosted reference tables

`../christy_bosted_{sp,dp}.txt.gz` are produced by `cb_reference.f`, linked against the
FORTRAN `F1F209.f` of P. E. Bosted and M. E. Christy (the copy in
`physics-code/Tools/F1F209_test.f`). The `sp` table uses the file as is; this is what
physics-code links. In the `dp` table, the single-precision literals of the original are
promoted to double, which allows a machine-precision comparison with the port.

```bash
F=~/Git/physics-code/Tools/F1F209_test.f
# as shipped (single-precision literals)
gfortran -O0 -std=legacy -w cb_reference.f $F -o cbref_sp && ./cbref_sp | gzip > ../christy_bosted_sp.txt.gz
# double-precision literals. -fdefault-double-8 keeps 1.d0 at 8 bytes, and FITEMC's REAL*4
# declaration must follow the promotion, otherwise the argument types mismatch
sed 's/REAL\*4 ALPHA, C,LN_C,X,A ,X_U/REAL ALPHA, C,LN_C,X,A ,X_U/' $F > F1F209_dbl.f
gfortran -O0 -std=legacy -w -fdefault-real-8 -fdefault-double-8 cb_reference.f F1F209_dbl.f -o cbref_dp \
  && ./cbref_dp | gzip > ../christy_bosted_dp.txt.gz
```

The grid is W = 0.9503 + 0.01 i GeV (i = 0..225) × 13 values of Q² from 0 to 8 GeV². The
0.3 MeV offset keeps every point away from the 1 MeV W-bin edges of `RESMODD`. The column
layout is documented at the top of `cb_reference.f` and in `tests/test_christy_bosted.py`.

## PRELIMINARY 2021 fit (F1F221)

`../christy_bosted21_{sp,dp}.txt.gz` are produced by `cb21_reference.f`, linked against
`F1F2IN21_v1.0.f` ("F1F221 v1.0 prerelease 1", M. E. Christy). That file comes from JLab
SIMC, https://github.com/JeffersonLab/simc_gfortran, which carries no license, so it is not
copied here. The driver must run in a directory that contains SIMC's `f1f2tables/`: the
on- and off-shell CD-Bonn tables plus `cdbn.qwave`, which the quasi-elastic
initialization `SQESUB(..., first=.true.)` reads. SIMC itself calls `SQESUB` once before
any `F1F2IN21` call, and so does the driver.

```bash
B=https://raw.githubusercontent.com/JeffersonLab/simc_gfortran/master
curl -sfLO $B/F1F2IN21_v1.0.f; mkdir -p f1f2tables
for f in f11 f12 f22; do for s in onshell offshell; do
  curl -sfL -o f1f2tables/$f-$s-cdbonn.dat $B/f1f2tables/$f-$s-cdbonn.dat; done; done
curl -sfL -o f1f2tables/cdbn.qwave $B/f1f2tables/cdbn.qwave
FL="-O0 -std=legacy -w -ffixed-line-length-none"
# as shipped. The single-precision build needs one type fix in the quasi-elastic code,
# which is not used for the inelastic output.
sed 's/sign(1\.,(a-b))/sign(1.d0,(a-b))/' F1F2IN21_v1.0.f > F1F2IN21_v1.0_signfix.f
gfortran $FL cb21_reference.f F1F2IN21_v1.0_signfix.f -o cb21_sp && ./cb21_sp | gzip > christy_bosted21_sp.txt.gz
# double-precision literals
gfortran $FL -fdefault-real-8 -fdefault-double-8 cb21_reference.f F1F2IN21_v1.0.f -o cb21_dp \
  && ./cb21_dp | gzip > christy_bosted21_dp.txt.gz
```

The grid is the same W grid × 16 values of Q² from 0 to 25 GeV², including 0.049 and 0.05
around the switch between the two smearing branches. At Q² = 0, F2 of p and n is NaN in
the Fortran (0/0); the port returns 0 there.
