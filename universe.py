"""
Universe: N-body simulation engine.

Force selection, RK4 and leapfrog integration, and conservation diagnostics.
"""

import numpy as np
from core import Body, MultipoleExpansion, ParticleMesh, _finite_scalar


class Universe:
    FORCE_METHODS = ("direct", "barnes-hut", "multipole", "particle-mesh")
    INTEGRATORS = ("rk4", "leapfrog")

    def __init__(self, dt: float = 0.1, G: float = 6.6743e-11,
                 epsilon: float = 1e-3, theta: float = 0.5,
                 force_method="direct", *, multipole_order=2, grid_size=64,
                 box_size=None, box_origin=None, record_every=1):
        """
        Parameters
        dt, G :          Positive timestep and gravitational constant.
        epsilon :        Plummer softening length (unused by particle-mesh).
        theta :          Tree opening angle; zero gives direct summation.
        box_size :       Fixed periodic box side, or fit once if omitted.
        record_every :   Save every N steps plus endpoints; 0 disables recording.

        Isolated energy diagnostics cost O(N²), including for tree runs.
        Particle-mesh uses a periodic grid energy diagnostic instead.
        """
        self.bodies: list[Body] = []
        self.dt = _finite_scalar(dt, "dt", positive=True)
        self.G = _finite_scalar(G, "G", positive=True)
        self.epsilon = _finite_scalar(epsilon, "epsilon")
        self.theta = _finite_scalar(theta, "theta")
        MultipoleExpansion(self.theta, multipole_order)
        self.multipole_order = multipole_order
        self.particle_mesh = ParticleMesh(grid_size, box_size, origin=box_origin)
        self.force_method = force_method
        if (isinstance(record_every, (bool, np.bool_))
                or not isinstance(record_every, (int, np.integer))
                or record_every < 0):
            raise ValueError("record_every must be a non-negative integer")
        self.record_every = int(record_every)
        self.time = 0.0
        self.steps = 0

        # conservation diagnostics
        self.diag_time: list[float] = []
        self.diag_KE: list[float] = []
        self.diag_PE: list[float] = []
        self.diag_E: list[float] = []
        self.diag_L: list[np.ndarray] = []
        self.diag_P: list[np.ndarray] = []

    @property
    def force_method(self):
        return self._force_method

    @force_method.setter
    def force_method(self, value):
        if value not in self.FORCE_METHODS:
            raise ValueError(f"Unknown force_method {value!r}; choose from {self.FORCE_METHODS}")
        self._force_method = value

    @property
    def energy_description(self):
        if self.force_method == "particle-mesh":
            return "periodic mesh energy (includes mesh self-energy; grid diagnostic)"
        return "isolated Plummer pair energy (exact diagnostic; approximate tree forces)"

    # body management
    def add_body(self, body: Body):
        if not isinstance(body, Body):
            raise TypeError("add_body expects a Body")
        if any(existing is body for existing in self.bodies):
            raise ValueError("The same Body cannot be added twice")
        self.bodies.append(body)

    # force computation
    def _compute_accelerations_direct(self):
        """Pairwise Plummer gravity with equal-and-opposite pair forces."""
        for body in self.bodies:
            body.acceleration[:] = 0
        for i, first in enumerate(self.bodies):
            for second in self.bodies[i + 1:]:
                delta = second.position - first.position
                d2 = delta @ delta + self.epsilon**2
                if d2 == 0:
                    raise ValueError("Coincident particles require epsilon > 0")
                kernel = self.G * delta / d2**1.5
                first.acceleration += second.mass * kernel
                second.acceleration -= first.mass * kernel

    def _compute_accelerations_barneshut(self):
        MultipoleExpansion(self.theta, order=0).compute_accelerations(
            self.bodies, self.G, self.epsilon)

    def compute_accelerations(self):
        if self.force_method == "direct":
            self._compute_accelerations_direct()
        elif self.force_method == "barnes-hut":
            self._compute_accelerations_barneshut()
        elif self.force_method == "multipole":
            MultipoleExpansion(self.theta, self.multipole_order).compute_accelerations(
                self.bodies, self.G, self.epsilon)
        else:
            self.particle_mesh.compute_accelerations(self.bodies, self.G, self.epsilon)

    def _start_step(self):
        if self.record_every and not self.diag_time:
            self._record()

    def _finish_step(self):
        self.time += self.dt
        self.steps += 1
        if self.record_every and self.steps % self.record_every == 0:
            self._record()

    # integrators
    def step_rk4(self):
        """Classical fixed-step RK4; four stages plus final acceleration refresh."""
        self._start_step()
        dt = self.dt
        r0 = np.array([b.position for b in self.bodies]).reshape(-1, 3)
        v0 = np.array([b.velocity for b in self.bodies]).reshape(-1, 3)

        def acceleration_at(positions):
            for body, pos in zip(self.bodies, positions):
                body.position[:] = pos
            self.compute_accelerations()
            return np.array([b.acceleration for b in self.bodies]).reshape(-1, 3)

        # k1
        k1r = v0
        k1v = acceleration_at(r0)

        # k2, k3: midpoint estimates
        k2r = v0 + 0.5 * dt * k1v
        k2v = acceleration_at(r0 + 0.5 * dt * k1r)
        k3r = v0 + 0.5 * dt * k2v
        k3v = acceleration_at(r0 + 0.5 * dt * k2r)

        # k4: full step
        k4r = v0 + dt * k3v
        k4v = acceleration_at(r0 + dt * k3r)

        # weighted combination
        positions = r0 + dt / 6 * (k1r + 2*k2r + 2*k3r + k4r)
        velocities = v0 + dt / 6 * (k1v + 2*k2v + 2*k3v + k4v)
        for body, pos, vel in zip(self.bodies, positions, velocities):
            body.position[:] = pos
            body.velocity[:] = vel
        self.compute_accelerations()
        self._finish_step()

    def step_leapfrog(self):
        """Second-order kick-drift-kick; symplectic for conservative forces.

        One-sided tree approximations and the PM grid force do not guarantee
        the same Hamiltonian conservation properties as exact pair gravity.
        """
        self._start_step()

        # kick (half), then drift (full)
        self.compute_accelerations()
        for body in self.bodies:
            body.velocity += 0.5 * self.dt * body.acceleration
            body.position += self.dt * body.velocity

        # kick (half) at the new positions
        self.compute_accelerations()
        for body in self.bodies:
            body.velocity += 0.5 * self.dt * body.acceleration
        self._finish_step()

    # conservation diagnostics
    def kinetic_energy(self) -> float:
        return float(sum(0.5 * b.mass * (b.velocity @ b.velocity) for b in self.bodies))

    def potential_energy(self) -> float:
        if self.force_method == "particle-mesh":
            return self.particle_mesh.potential_energy(self.bodies, self.G)
        energy = 0.0
        for i, first in enumerate(self.bodies):
            for second in self.bodies[i + 1:]:
                delta = second.position - first.position
                distance = np.sqrt(delta @ delta + self.epsilon**2)
                if distance == 0:
                    raise ValueError("Coincident particles require epsilon > 0")
                energy -= self.G * first.mass * second.mass / distance
        return float(energy)

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def linear_momentum(self) -> np.ndarray:
        return sum((b.mass * b.velocity for b in self.bodies), np.zeros(3))

    def angular_momentum(self) -> np.ndarray:
        return sum((b.mass * np.cross(b.position, b.velocity) for b in self.bodies), np.zeros(3))

    def _record(self):
        ke = self.kinetic_energy()
        pe = self.potential_energy()
        for body in self.bodies:
            body.snapshot()
        self.diag_time.append(self.time)
        self.diag_KE.append(ke)
        self.diag_PE.append(pe)
        self.diag_E.append(ke + pe)
        self.diag_L.append(self.angular_momentum().copy())
        self.diag_P.append(self.linear_momentum().copy())

    # run helper
    def run(self, steps: int, method: str = "leapfrog", progress: bool = True):
        """Advance the simulation by the requested number of timesteps."""
        if method not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator {method!r}; choose from {self.INTEGRATORS}")
        if (isinstance(steps, (bool, np.bool_))
                or not isinstance(steps, (int, np.integer))
                or steps < 0):
            raise ValueError("steps must be a non-negative integer")
        stepper = self.step_rk4 if method == "rk4" else self.step_leapfrog
        for i in range(steps):
            stepper()
            if progress and (i + 1) % max(1, steps // 10) == 0:
                pct = 100 * (i + 1) / steps
                print(f"  [{pct:5.1f}%]  t = {self.time:.6e}")
        if steps and self.record_every and self.diag_time[-1] != self.time:
            self._record()
