"""Physical constants used throughout the package.

Conventions
-----------
* All energies, momenta and masses are in **GeV** (the C++ code in
  ``physics-code`` works in MeV; values below are converted).
* The deuteron/nucleon kinematics uses the *average* nucleon mass
  m = (m_n + m_p)/2, as in the paper (Sec. II A) and in the C++ ``MASSn``.
"""

import math

#: proton mass [GeV] (same value as physics-code ``MASSP``)
M_PROTON = 0.93827231
#: neutron mass [GeV] (same value as physics-code ``MASSN``)
M_NEUTRON = 0.93956536
#: average nucleon mass m = (m_n + m_p)/2 [GeV]; used in the LF kinematics
M_NUCLEON = 0.5 * (M_PROTON + M_NEUTRON)
#: deuteron mass [GeV] (same value as physics-code ``MASSD``)
M_DEUTERON = 1.8756

#: hbar*c [GeV fm]  (TWavefunctionImplementation::fgHbarc = 197.326968 MeV fm)
HBARC = 0.197326968
#: (hbar c)^2 in nb GeV^2 : 1 GeV^-2 = 0.3893794 mb = 3.893794e5 nb
GEV2_TO_NB = 3.893794e5

#: fine-structure constant (physics-code ``ALPHA``)
ALPHA_EM = 0.00729735253

PI = math.pi
#: the factor [2(2pi)^3] that appears in the tagged structure functions and in
#: the invariant phase space element dGamma_p, Eq. (2.15a). It cancels in the
#: cross section; we keep it explicit to follow the paper's conventions.
TWO_2PI3 = 2.0 * (2.0 * math.pi) ** 3
