# Deuteron smearing tables (PRELIMINARY Christy 2021 fit)

These are the light-cone momentum-distribution smearing functions f11, f12 and f22 for
the deuteron in the weak-binding approximation, computed with the CD-Bonn wave function.
`taggedpd.christy_bosted21` uses them. The files are gzipped, otherwise unchanged copies
of `f1f2tables/f{11,12,22}-onshell-cdbonn.dat` from the Jefferson Lab SIMC repository,
https://github.com/JeffersonLab/simc_gfortran. They accompany `F1F2IN21_v1.0.f`
(M. E. Christy, "F1F221 v1.0 prerelease 1").

Each file has three columns, gamma, y and f(gamma, y): 21 gamma values (1.0 to 5.0 in
steps of 0.2) times 999 y values (0.002 to 1.998).

The SIMC repository has no license file, and permission to redistribute these tables
has not been confirmed. The parametrization they belong to is preliminary.
