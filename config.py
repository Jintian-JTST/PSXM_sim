"""Adopted working assumptions of the design study.

The report repeatedly flags these as study assumptions rather than
beam- or hardware-derived limits (report sections 3, 6.2 and 8), so they
are collected here in one place that every script reads from.
"""

MAX_CURRENT = 1000.0   # A, per-coil current cap
G = 1e-3               # T/mm, quadrupole central-gradient benchmark (G*)
DIPOLE_TARGET = 1e-3   # T, provisional dipole central-field benchmark
SHIELD_N = 200         # shield discretisation points (baseline)
SAMPLE_GAP_MM = 5.0    # radial gap: shield currents -> B=0 sample ring
OUTER_MM = 2.0         # second B=0 ring, this far outside the shield
N_BETWEEN = 3          # B=0 samples between adjacent shield points
