"""
Core data structures and gravity calculations for the N-body solver.

Body, octree, multipole and particle-mesh forces.
"""

import itertools
import numpy as np


def _finite_scalar(value, name, *, positive=False):
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {bound}")
    return value


class Body:
    """Point mass with position, velocity and history tracking."""

    __slots__ = (
        "name", "color", "mass", "position", "velocity",
        "acceleration", "history_pos", "history_vel",
    )

    def __init__(self,
                 name: str,
                 mass: float,
                 position: list | np.ndarray,
                 velocity: list | np.ndarray,
                 color: str = "white"):
        self.name = name
        self.color = color
        self.mass = _finite_scalar(mass, "mass", positive=True)

        # Copy the inputs so bodies do not share state vectors.
        self.position = np.array(position, dtype=np.float64, copy=True)
        self.velocity = np.array(velocity, dtype=np.float64, copy=True)
        for name, vector in (("position", self.position), ("velocity", self.velocity)):
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite three-vector")
        self.acceleration = np.zeros(3)
        self.history_pos: list[np.ndarray] = []
        self.history_vel: list[np.ndarray] = []

    def snapshot(self):
        """Save the current state to history."""
        self.history_pos.append(self.position.copy())
        self.history_vel.append(self.velocity.copy())

    def __repr__(self):
        return f"Body({self.name}, m={self.mass:.3e})"


# Barnes-Hut octree

class OctreeNode:
    """Octree cell storing mass, center of mass and second moment S = sum(m s s^T).

    Leaves can hold several coincident bodies. The depth limit also prevents
    endless subdivision when positions are nearly identical.
    """

    __slots__ = (
        "center", "half_width", "mass", "com", "moment",
        "bodies", "children", "is_leaf", "depth",
    )

    def __init__(self, center, half_width, depth=0):
        self.center = np.array(center, dtype=float, copy=True)
        self.half_width = float(half_width)
        self.mass = 0.0
        self.com = np.zeros(3)
        self.moment = np.zeros((3, 3))
        self.bodies = []
        self.children = [None] * 8
        self.is_leaf = True
        self.depth = depth

    @property
    def body(self):
        """Compatibility view for a single-particle leaf."""
        return self.bodies[0] if len(self.bodies) == 1 else None

    @property
    def quadrupole(self):
        return 3 * self.moment - np.trace(self.moment) * np.eye(3)

    # octant index mapping
    def _octant(self, pos: np.ndarray):
        idx = 0
        if pos[0] >= self.center[0]:
            idx |= 1
        if pos[1] >= self.center[1]:
            idx |= 2
        if pos[2] >= self.center[2]:
            idx |= 4
        return idx

    def _child_center(self, octant):
        signs = np.array([1 if octant & (1 << axis) else -1 for axis in range(3)])
        return self.center + signs * self.half_width * 0.5

    # tree construction
    def _insert_into_octant(self, body: Body):
        octant = self._octant(body.position)
        if self.children[octant] is None:
            self.children[octant] = OctreeNode(
                self._child_center(octant), self.half_width * 0.5, self.depth + 1)
        self.children[octant].insert(body)

    def insert(self, body: Body):
        # Update mass and moments using the parallel-axis theorem.
        total = self.mass + body.mass
        delta = body.position - self.com
        self.moment += self.mass * (body.mass / total) * np.outer(delta, delta)
        self.com += (body.mass / total) * delta
        self.mass = total

        if self.is_leaf:
            # Keep coincident bodies together instead of subdividing forever.
            if (not self.bodies or self.depth >= 48 or
                    np.array_equal(body.position, self.bodies[0].position)):
                self.bodies.append(body)
                return
            existing = self.bodies
            self.bodies = []
            self.is_leaf = False
            for previous in existing:
                self._insert_into_octant(previous)
        self._insert_into_octant(body)

    # force computation
    def compute_acceleration(self, body: Body, theta: float, G: float,
                             epsilon: float, order: int = 0):
        """Tree walk with monopole (0) or quadrupole (2) expansion.

        All terms expand the Plummer kernel 1/sqrt(r² + epsilon²).
        A cell containing the target is always opened to exclude self-force.
        theta=0 opens every cell and recovers direct summation.
        """
        if self.mass == 0:
            return np.zeros(3)
        if self.is_leaf:
            acc = np.zeros(3)
            for source in self.bodies:
                if source is body:
                    continue
                delta = source.position - body.position
                d2 = delta @ delta + epsilon**2
                if d2 == 0:
                    raise ValueError("Coincident particles require epsilon > 0")
                acc += G * source.mass * delta / d2**1.5
            return acc

        r = body.position - self.com
        distance = np.linalg.norm(r)
        contains_target = np.all(np.abs(body.position - self.center) <= self.half_width)
        if not contains_target and 2 * self.half_width < theta * distance:
            d2 = r @ r + epsilon**2
            acc = -G * self.mass * r / d2**1.5
            if order == 2:
                # Retaining trace(S) is necessary for the softened kernel.
                Sr = self.moment @ r
                acc += G * (3 * Sr + 1.5 * np.trace(self.moment) * r) / d2**2.5
                acc -= 7.5 * G * (r @ Sr) * r / d2**3.5
            return acc
        acc = np.zeros(3)
        for child in self.children:
            if child is not None:
                acc += child.compute_acceleration(body, theta, G, epsilon, order)
        return acc


def build_octree(bodies: list[Body]) -> OctreeNode:
    """Build a padded bounding cube around the current particle positions."""
    if not bodies:
        return OctreeNode(np.zeros(3), 1.0)
    positions = np.array([b.position for b in bodies])
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center = lo + 0.5 * (hi - lo)
    padding = max(float(np.max(np.abs(positions))) * np.finfo(float).eps * 8, 1e-15)
    half_width = float(np.max(hi - lo)) * 0.505 + padding
    tree = OctreeNode(center, half_width)
    for body in bodies:
        tree.insert(body)
    return tree


# Multipole expansion

class MultipoleExpansion:
    """Quadrupole tree code, typically O(N log N); not a full linear-time FMM."""

    def __init__(self, theta=0.5, order=2):
        self.theta = _finite_scalar(theta, "theta")
        if order not in (0, 2):
            raise ValueError("multipole order must be 0 (monopole) or 2 (quadrupole)")
        self.order = order

    def compute_accelerations(self, bodies, G, epsilon):
        tree = build_octree(bodies)
        for body in bodies:
            body.acceleration[:] = tree.compute_acceleration(
                body, self.theta, G, epsilon, self.order)


# Particle-mesh gravity

class ParticleMesh:
    """Periodic CIC/FFT Poisson solver, O(N + M log M), M=grid_size³.

    The zero Fourier mode removes the uniform background. CIC deposition and
    interpolation use the same weights. Forces use centered grid differences.
    Positions wrap modulo a fixed box; stored trajectories stay unwrapped.
    epsilon is unused: the grid spacing sets the force resolution.

    Supply box_size for reproducible periodic experiments. If omitted, a box
    is fit once at the first nonempty evaluation and then retained. Padding
    reduces image effects but never changes these into isolated boundaries.
    """

    def __init__(self, grid_size=64, box_size=None, pad=0.5, origin=None):
        if (isinstance(grid_size, (bool, np.bool_))
                or not isinstance(grid_size, (int, np.integer))
                or grid_size < 4):
            raise ValueError("grid_size must be an integer >= 4")
        self.grid_size = int(grid_size)
        self.box_size = box_size
        if box_size is not None:
            self.box_size = _finite_scalar(box_size, "box_size", positive=True)
        self.pad = _finite_scalar(pad, "pad")
        if origin is not None and box_size is None:
            raise ValueError("origin requires an explicit box_size")
        self.origin = None
        if origin is not None:
            self.origin = np.array(origin, dtype=float, copy=True)
            if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
                raise ValueError("origin must be a finite three-vector")
        if self.box_size is not None and self.origin is None:
            self.origin = np.full(3, -0.5 * self.box_size)

    def _ensure_box(self, positions):
        if self.box_size is None:
            lo, hi = positions.min(axis=0), positions.max(axis=0)
            span = float(np.max(hi - lo))
            self.box_size = span * (1 + 2 * self.pad) if span > 0 else 1.0
            self.origin = lo + 0.5 * (hi - lo) - 0.5 * self.box_size

    def _cic(self, positions):
        """Grid corners and cloud-in-cell weights for each particle."""
        self._ensure_box(positions)
        n = self.grid_size
        x = np.remainder((positions - self.origin) / self.box_size, 1.0) * n
        base = np.floor(x).astype(np.int64)
        fraction = x - base
        corners = []
        for offset in itertools.product((0, 1), repeat=3):
            offset = np.array(offset)
            indices = (base + offset) % n
            weights = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            corners.append((tuple(indices.T), weights))
        return corners

    @staticmethod
    def _interpolate(field, corners):
        return sum(field[indices] * weights for indices, weights in corners)

    def _solve(self, bodies, G):
        positions = np.array([b.position for b in bodies])
        masses = np.array([b.mass for b in bodies])
        corners = self._cic(positions)
        n = self.grid_size
        h = self.box_size / n
        rho = np.zeros((n, n, n))
        for indices, weights in corners:
            np.add.at(rho, indices, masses * weights / h**3)
        phi = self._solve_density(rho, G)
        return rho, phi, corners

    def _solve_density(self, rho, G):
        """Solve ∇²phi = 4 pi G (rho - mean(rho)) on the periodic grid."""
        n = self.grid_size
        h = self.box_size / n
        k = 2 * np.pi * np.fft.fftfreq(n, d=h)
        kz = 2 * np.pi * np.fft.rfftfreq(n, d=h)
        k2 = k[:, None, None]**2 + k[None, :, None]**2 + kz[None, None, :]**2
        k2[0, 0, 0] = 1.0
        phi_hat = -4 * np.pi * G * np.fft.rfftn(rho) / k2
        phi_hat[0, 0, 0] = 0.0
        return np.fft.irfftn(phi_hat, s=rho.shape, axes=(0, 1, 2))

    def compute_accelerations(self, bodies, G, epsilon=0.0):
        if not bodies:
            return
        _, phi, corners = self._solve(bodies, G)
        h = self.box_size / self.grid_size
        components = []
        for axis in range(3):
            gradient = (np.roll(phi, -1, axis) - np.roll(phi, 1, axis)) / (2 * h)
            components.append(self._interpolate(-gradient, corners))
        acceleration = np.column_stack(components)
        for body, acc in zip(bodies, acceleration):
            body.acceleration[:] = acc

    def potential_energy(self, bodies, G):
        """Periodic grid diagnostic 1/2 ∫rho phi dV, including mesh self-energy.

        It is not the isolated pair potential, nor an exact Hamiltonian for
        centered-difference/CIC forces. Grid error contributes to its drift.
        """
        if not bodies:
            return 0.0
        rho, phi, _ = self._solve(bodies, G)
        return float(0.5 * np.sum(rho * phi) * (self.box_size / self.grid_size)**3)
