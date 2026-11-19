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
