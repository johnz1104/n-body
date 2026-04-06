"""
Universe: simulation engine
 
Supports:
  • Direct O(N²) pairwise gravity
  • Barnes-Hut O(N log N) approximate gravity
  • RK4 and symplectic leapfrog integrators
  • Energy / momentum conservation tracking
"""
 
import numpy as np
from typing import Literal
from core import Body, OctreeNode, build_octree
 
 
class Universe:
 
    def __init__(self, dt: float = 0.1, G: float = 6.6743e-11,
                 epsilon: float = 1e-3, theta: float = 0.5,
                 force_method: Literal["direct", "barnes-hut"] = "direct"):
        """
        Parameters
        dt : float       Time-step (seconds).
        G  : float       Gravitational constant.
        epsilon : float  Softening length (same units as positions).
        theta : float    Barnes-Hut opening angle (ignored for 'direct').
        force_method :   'direct' for O(N²), 'barnes-hut' for O(N log N).
        """
        self.bodies: list[Body] = []
        self.G = G
        self.dt = dt
        self.epsilon = epsilon
        self.theta = theta
        self.force_method = force_method
        self.time = 0.0
 
        # conservation diagnostics — populated every step
        self.diag_time: list[float] = []
        self.diag_KE: list[float] = []
        self.diag_PE: list[float] = []
        self.diag_E: list[float] = []
        self.diag_L: list[np.ndarray] = []      # angular momentum vector
        self.diag_P: list[np.ndarray] = []      # linear momentum vector
 
    # body management
    def add_body(self, body: Body):
        self.bodies.append(body)
 
    # force computation
    def _compute_accelerations_direct(self):
        """O(N²) pairwise Newtonian gravity."""
        for b in self.bodies:
            b.acceleration[:] = 0.0
 
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = self.bodies[i], self.bodies[j]
                r_vec = b2.position - b1.position
                dist = np.linalg.norm(r_vec)
                soft = dist + self.epsilon
                force_over_soft = self.G * b1.mass * b2.mass / (soft * soft * soft)
                # Newton III: equal and opposite
                acc_ij = force_over_soft * r_vec
                b1.acceleration += acc_ij / b1.mass
                b2.acceleration -= acc_ij / b2.mass
 
    def _compute_accelerations_barneshut(self):
        """Barnes-Hut O(N log N) approximate gravity."""
        tree = build_octree(self.bodies)
        for b in self.bodies:
            b.acceleration = tree.compute_acceleration(
                b, self.theta, self.G, self.epsilon
            )
 
    def compute_accelerations(self):
        if self.force_method == "barnes-hut":
            self._compute_accelerations_barneshut()
        else:
            self._compute_accelerations_direct()
 
    # integrators
    def step_rk4(self):
        """Classical 4th-order Runge-Kutta."""
        dt = self.dt
        n = len(self.bodies)
 
        # cache initial state
        r0 = [b.position.copy() for b in self.bodies]
        v0 = [b.velocity.copy() for b in self.bodies]
 
        # k1
        self.compute_accelerations()
        k1r = [b.velocity.copy() for b in self.bodies]
        k1v = [b.acceleration.copy() for b in self.bodies]
 
        # k2 — evaluate at midpoint using k1
        for i in range(n):
            self.bodies[i].position = r0[i] + 0.5 * dt * k1r[i]
            self.bodies[i].velocity = v0[i] + 0.5 * dt * k1v[i]
        self.compute_accelerations()
        k2r = [b.velocity.copy() for b in self.bodies]
        k2v = [b.acceleration.copy() for b in self.bodies]
 
        # k3 — evaluate at midpoint using k2
        for i in range(n):
            self.bodies[i].position = r0[i] + 0.5 * dt * k2r[i]
            self.bodies[i].velocity = v0[i] + 0.5 * dt * k2v[i]
        self.compute_accelerations()
        k3r = [b.velocity.copy() for b in self.bodies]
        k3v = [b.acceleration.copy() for b in self.bodies]
 
        # k4 — evaluate at end using k3
        for i in range(n):
            self.bodies[i].position = r0[i] + dt * k3r[i]
            self.bodies[i].velocity = v0[i] + dt * k3v[i]
        self.compute_accelerations()
        k4r = [b.velocity.copy() for b in self.bodies]
        k4v = [b.acceleration.copy() for b in self.bodies]
 
        # weighted combination
        for i in range(n):
            b = self.bodies[i]
            b.position = r0[i] + (dt / 6.0) * (k1r[i] + 2*k2r[i] + 2*k3r[i] + k4r[i])
            b.velocity = v0[i] + (dt / 6.0) * (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i])
 
        self.time += dt
        self._record()
 
    def step_leapfrog(self):
        """Symplectic Kick-Drift-Kick leapfrog."""
        dt = self.dt
 
        # kick (half)
        self.compute_accelerations()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.acceleration
 
        # drift (full)
        for b in self.bodies:
            b.position += dt * b.velocity
 
        # kick (half)
        self.compute_accelerations()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.acceleration
 
        self.time += dt
        self._record()
 
    # conservation diagnostics
    def kinetic_energy(self) -> float:
        return sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in self.bodies)
 
    def potential_energy(self) -> float:
        pe = 0.0
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.bodies[j].position - self.bodies[i].position)
                pe -= self.G * self.bodies[i].mass * self.bodies[j].mass / (r + self.epsilon)
        return pe
 
    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()
 
    def linear_momentum(self) -> np.ndarray:
        return sum(b.mass * b.velocity for b in self.bodies)
 
    def angular_momentum(self) -> np.ndarray:
        return sum(b.mass * np.cross(b.position, b.velocity) for b in self.bodies)
 
    def _record(self):
        """Snapshot positions/velocities and conservation quantities."""
        for b in self.bodies:
            b.snapshot()
        ke = self.kinetic_energy()
        pe = self.potential_energy()
        self.diag_time.append(self.time)
        self.diag_KE.append(ke)
        self.diag_PE.append(pe)
        self.diag_E.append(ke + pe)
        self.diag_L.append(self.angular_momentum().copy())
        self.diag_P.append(self.linear_momentum().copy())
 
    # run helper
    def run(self, steps: int,
            method: Literal["rk4", "leapfrog"] = "leapfrog",
            progress: bool = True):
        """
        Advance the simulation by *steps* time-steps.
        """
        stepper = self.step_rk4 if method == "rk4" else self.step_leapfrog
        for i in range(steps):
            stepper()
            if progress and (i + 1) % max(1, steps // 10) == 0:
                pct = 100 * (i + 1) / steps
                print(f"  [{pct:5.1f}%]  t = {self.time:.3e} s  "
                      f"E = {self.diag_E[-1]:.6e} J")
