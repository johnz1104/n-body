"""
Convert N-body trajectories to radial velocities and sky-plane positions.

Observer geometry is fixed. Projections and interpolation retain JAX gradients.
"""

import numpy as np
from core import _jax_modules, _concrete_array, _finite_scalar


def _unit_vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"{name} must have a finite nonzero norm")
    return vector / norm


def _body_index(index, n):
    if (isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < n):
        raise ValueError(f"body index must be an integer in [0, {n})")
    return int(index)


# Observer projections

def radial_velocity(trajectory, body_index=0, *, line_of_sight=(0, 0, 1),
                    systemic_velocity=0.0):
    """Line-of-sight velocity, positive away from the observer.

    line_of_sight points from the observer toward the system and is normalized.
    Units match the simulation. systemic_velocity can be a fit parameter.
    """
    _, jnp = _jax_modules()
    index = _body_index(body_index, trajectory.velocities.shape[1])
    los = jnp.asarray(_unit_vector(line_of_sight, "line_of_sight"))
    return trajectory.velocities[:, index, :] @ los + systemic_velocity


def sky_plane_positions(trajectory, body_index=0, *, reference_index=None,
                        line_of_sight=(0, 0, 1), up=(0, 1, 0), distance=None):
    """Project positions onto two perpendicular sky axes, returning (T, 2).

    reference_index selects a body to measure offsets from. With distance in
    simulation length units, results are small-angle offsets in radians;
    otherwise they remain lengths. No parallax or light-travel time correction.
    """
    _, jnp = _jax_modules()
    n = trajectory.positions.shape[1]
    index = _body_index(body_index, n)
    los = _unit_vector(line_of_sight, "line_of_sight")
    up = _unit_vector(up, "up")
    right = np.cross(up, los)
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("up must not be parallel to line_of_sight")
    right /= np.linalg.norm(right)
    sky_up = np.cross(los, right)
    basis = jnp.asarray(np.stack((right, sky_up), axis=1))
    positions = trajectory.positions[:, index, :]
    if reference_index is not None:
        reference = _body_index(reference_index, n)
        positions = positions - trajectory.positions[:, reference, :]
    result = positions @ basis
    if distance is not None:
        distance = _finite_scalar(distance, "distance", positive=True)
        result = result / distance
    return result


# Observation times

def sample_observable(trajectory, values, times):
    """Linearly interpolate (T, ...) values at observation times (K,).

    Out-of-range times raise ValueError outside jit and return NaN inside it.
    Check interpolation error when choosing the simulation timestep.
    """
    jax, jnp = _jax_modules()
    values = jnp.asarray(values, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    grid = trajectory.times
    if values.ndim < 1 or values.shape[0] != grid.size or grid.size == 0:
        raise ValueError("values must have leading dimension equal to trajectory times")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("observation times must be a nonempty vector")
    requested = _concrete_array(times)
    known = _concrete_array(grid)
    if requested is not None and not np.isfinite(requested).all():
        raise ValueError("observation times must be finite")
    if known is not None:
        if not np.isfinite(known).all() or np.any(np.diff(known) <= 0):
            raise ValueError("trajectory times must be finite and strictly increasing")
        if requested is not None and np.any((requested < known[0]) | (requested > known[-1])):
            raise ValueError("observation times must lie inside the simulated interval")

    def interpolate(column):
        return jnp.interp(times, grid, column, left=jnp.nan, right=jnp.nan)

    # Apply the same interpolation to each observable component.
    flat = values.reshape(grid.size, -1)
    sampled = jax.vmap(interpolate, in_axes=1, out_axes=1)(flat)
    return sampled.reshape((times.size,) + values.shape[1:])
