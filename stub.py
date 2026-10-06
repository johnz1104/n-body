"""Compatibility imports. New code should select forces through Universe.

FastMultipole is a legacy name for the quadrupole tree code, not full FMM.
"""

from core import MultipoleExpansion, ParticleMesh


class FastMultipole(MultipoleExpansion):
    """Legacy alias for MultipoleExpansion."""


def attach(universe, backend):
    """Configure built-in forces without replacing Universe methods."""
    if isinstance(backend, ParticleMesh):
        universe.force_method = "particle-mesh"
        universe.particle_mesh = backend
    elif isinstance(backend, MultipoleExpansion):
        universe.force_method = "multipole"
        universe.theta = backend.theta
        universe.multipole_order = backend.order
    else:
        raise TypeError("backend must be ParticleMesh or MultipoleExpansion")
    return universe
