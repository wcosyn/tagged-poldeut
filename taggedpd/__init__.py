"""taggedpd -- polarized deuteron DIS with spectator nucleon tagging (impulse approximation).

Lightweight, numpy-vectorized implementation of

  W. Cosyn and C. Weiss, "Semi-inclusive deep-inelastic scattering on a polarized
  spin-1 target. II. Deuteron and spectator nucleon tagging" (JLAB-THY-26-4661),

with the cross-section decomposition of Part I (same authors). Main entry points:

* :class:`~taggedpd.wavefunctions.DeuteronWavefunction` -- AV18 / CDBonn / Paris radial functions
* :class:`~taggedpd.densities.LFDistributions`          -- neutron LF momentum/spin distributions (Sec. IV)
* :mod:`~taggedpd.nucleon`                              -- nucleon F1, F2, g1, g2 providers
* :class:`~taggedpd.crosssection.TaggedDIS`             -- structure functions, sigma_red, dsigma, asymmetries
* :func:`~taggedpd.inclusive.inclusive_sf`               -- unpolarized inclusive SFs / sigma_T,L of p, n, d in (W, Q^2),
                                                           Q^2 = 0 to DIS; pluggable models (alias: photoproduction)
* :mod:`~taggedpd.christy_bosted`                       -- port of the Christy-Bosted F1F209 fits
* :mod:`~taggedpd.christy_bosted21`                     -- PRELIMINARY 2021 fit (port of SIMC F1F2IN21_v1.0.f)
"""

from .constants import M_DEUTERON, M_NUCLEON, M_NEUTRON, M_PROTON
from .wavefunctions import DeuteronWavefunction, available_wavefunctions
from .densities import LFDistributions, NeutronDistributions, cm_momentum, spectator_lf_from_rest_frame
from .nucleon import (NucleonStructure, SLACStructure, GRSV2000Structure, LHAPDFStructure,
                      CombinedStructure, GriddedStructure, default_structure)
from .crosssection import Kinematics, Polarization, TaggedDIS
from .inclusive import (inclusive_sf, photoproduction, InclusiveModel, PhotoabsorptionModel,
                        ChristyBosted, ChristyBosted2021, Piecewise, Blend, FromNucleonStructure,
                        AsNucleonStructure, ChristyBostedStructure, default_photoabsorption,
                        resonance_dis_model)

__version__ = "0.1.0"
