"""taggedpd -- polarized deuteron DIS with spectator nucleon tagging (impulse approximation).

Lightweight, numpy-vectorized implementation of

  W. Cosyn and C. Weiss, "Semi-inclusive deep-inelastic scattering on a polarized
  spin-1 target. II. Deuteron and spectator nucleon tagging" (JLAB-THY-26-4661),

with the cross-section decomposition of Part I (same authors). Main entry points:

* :class:`~taggedpd.wavefunctions.DeuteronWavefunction` -- AV18 / CDBonn / Paris radial functions
* :class:`~taggedpd.densities.LFDistributions`          -- neutron LF momentum/spin distributions (Sec. IV)
* :mod:`~taggedpd.nucleon`                              -- nucleon F1, F2, g1, g2 providers
* :class:`~taggedpd.crosssection.TaggedDIS`             -- structure functions, sigma_red, dsigma, asymmetries
"""

from .constants import M_DEUTERON, M_NUCLEON, M_NEUTRON, M_PROTON
from .wavefunctions import DeuteronWavefunction, available_wavefunctions
from .densities import LFDistributions, NeutronDistributions, cm_momentum, spectator_lf_from_rest_frame
from .nucleon import (NucleonStructure, SLACStructure, GRSV2000Structure, LHAPDFStructure,
                      CombinedStructure, GriddedStructure, default_structure)
from .crosssection import Kinematics, Polarization, TaggedDIS

__version__ = "0.1.0"
