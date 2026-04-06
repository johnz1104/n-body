"""
Additional force backends for the N-body solver.

Completes the set of four computation methods alongside
``core.py`` / ``universe.py`` (Direct O(N²), Barnes-Hut O(N log N)):

    • FastMultipole  — quadrupole multipole tree code, O(N log N)
    • ParticleMesh   — FFT-based PM Poisson solver, O(N + M log M)

Both backends expose a uniform interface::

    backend.compute_accelerations(bodies, G, epsilon) -> None

which writes each body's ``acceleration`` attribute in place, matching
the contract of ``Universe._compute_accelerations_direct`` and
``_compute_accelerations_barneshut``.  Use :func:`attach` to install a
backend on an existing ``Universe``; the resulting object works with
``visualization.plot_results`` and ``print_conservation_summary``
unchanged — they only read ``body.history_pos`` and ``universe.diag_*``,
both of which are still populated by Universe.step_rk4 / step_leapfrog.
"""

import numpy as np
from core import Body


# ─────────────────────────────────────────────────────────────────────────────
#  Attachment helper
# ─────────────────────────────────────────────────────────────────────────────

def attach(universe, backend):
    """Install *backend* as the force calculator of a ``Universe`` instance.

    After this call, ``universe.run(...)`` and the ``step_*`` methods use
    ``backend.compute_accelerations``.  The Universe's ``_record`` path
    still populates ``history_pos`` / ``diag_*``, so the visualization
    module continues to work with no modification.
    """
    universe._backend = backend
    universe.force_method = backend.__class__.__name__
    universe.compute_accelerations = lambda: backend.compute_accelerations(
        universe.bodies, universe.G, universe.epsilon)
    return universe


#  Fast Multipole (quadrupole tree code)
class FastMultipole:
    """Quadrupole multipole tree code.

    Builds an octree of particles and stores, at each internal node:

        • total mass                                   M
        • centre of mass                               R
        • reduced traceless quadrupole tensor about R  Q_ij

    The force on a target particle from a cell is approximated by its
    multipole expansion whenever the well-separated criterion
    ``cell_size / distance < theta`` is met; otherwise the walk recurses.
    Including the quadrupole term makes the expansion 2nd-order accurate
    in ``cell_size / distance``, meaningfully sharper than Barnes-Hut
    monopole-only for the same ``theta``.

    Complexity: O(N log N) per force evaluation.

    Note: this is the multipole-tree side of FMM.  A full linear-time
    FMM additionally translates cell multipoles into local expansions
    at every target cell (M2L + L2L), which is out of scope here.
    """

    def __init__(self, theta: float = 0.5):
        self.theta = float(theta)

    # tree node (plain dict to keep the file compact)
    @staticmethod
    def _make_node(center, half_width):
        return dict(
            center=center,
            half_width=half_width,
            mass=0.0,
            com=np.zeros(3),
            Q=np.zeros((3, 3)),
            children=[None] * 8,
            body=None,
            is_leaf=True,
        )

    @staticmethod
    def _octant(node, pos):
        idx = 0
        if pos[0] >= node["center"][0]: idx |= 1
        if pos[1] >= node["center"][1]: idx |= 2
        if pos[2] >= node["center"][2]: idx |= 4
        return idx

    @staticmethod
    def _child_center(node, octant):
        off = node["half_width"] * 0.5
        cx, cy, cz = node["center"]
        return np.array([
            cx + off if (octant & 1) else cx - off,
            cy + off if (octant & 2) else cy - off,
            cz + off if (octant & 4) else cz - off,
        ])

    def _insert(self, node, body):
        """Insert *body* into the subtree at *node*, updating M0 and COM."""
        if node["is_leaf"] and node["body"] is None and node["mass"] == 0.0:
            node["body"] = body
            node["mass"] = body.mass
            node["com"] = body.position.copy()
            return
        if node["is_leaf"] and node["body"] is not None:
            existing = node["body"]
            node["body"] = None
            node["is_leaf"] = False
            self._push_down(node, existing)
        self._push_down(node, body)
        total = node["mass"] + body.mass
        node["com"] = (node["com"] * node["mass"] + body.position * body.mass) / total
        node["mass"] = total

    def _push_down(self, node, body):
        o = self._octant(node, body.position)
        if node["children"][o] is None:
            node["children"][o] = self._make_node(
                self._child_center(node, o), node["half_width"] * 0.5)
        self._insert(node["children"][o], body)

    def _collect_leaves(self, node, out):
        if node is None:
            return
        if node["is_leaf"]:
            if node["body"] is not None:
                out.append(node["body"])
            return
        for c in node["children"]:
            self._collect_leaves(c, out)

    def _compute_quadrupole(self, node):
        """Fill in ``node.Q`` for every internal node.

        Q_ij = Σ_k m_k [3 s_ki s_kj − δ_ij |s_k|²]  with  s_k = x_k − R_com.
        Traceless by construction.
        """
        if node["is_leaf"]:
            node["Q"] = np.zeros((3, 3))
            return
        leaves = []
        self._collect_leaves(node, leaves)
        Q = np.zeros((3, 3))
        R = node["com"]
        I = np.eye(3)
        for b in leaves:
            s = b.position - R
            s2 = s @ s
            Q += b.mass * (3.0 * np.outer(s, s) - s2 * I)
        node["Q"] = Q
        for c in node["children"]:
            if c is not None:
                self._compute_quadrupole(c)

    def _walk(self, node, body, G, eps):
        """Return the acceleration on *body* from the subtree at *node*."""
        if node is None or node["mass"] == 0.0:
            return np.zeros(3)

        # r_vec points from the cell COM to the target body
        r_vec = body.position - node["com"]
        d2 = r_vec @ r_vec
        d = np.sqrt(d2)

        if node["is_leaf"]:
            if node["body"] is body or d == 0.0:
                return np.zeros(3)
            soft = d + eps
            # attractive monopole toward COM
            return -G * node["mass"] * r_vec / (soft * soft * soft)

        cell_size = 2.0 * node["half_width"]
        if cell_size < self.theta * d:
            # well separated → close-form multipole expansion
            soft = d + eps
            inv_d  = 1.0 / soft
            inv_d3 = inv_d * inv_d * inv_d
            inv_d5 = inv_d3 * inv_d * inv_d
            inv_d7 = inv_d5 * inv_d * inv_d

            # monopole: a = -G M r / d³
            a = -G * node["mass"] * r_vec * inv_d3

            # quadrupole:  a += G Q·r / d⁵  −  (5G/2)(rᵀ Q r) r / d⁷
            Q = node["Q"]
            Qr  = Q @ r_vec
            rQr = r_vec @ Qr
            a += G * Qr * inv_d5 - 2.5 * G * rQr * r_vec * inv_d7
            return a

        # otherwise recurse
        acc = np.zeros(3)
        for c in node["children"]:
            acc = acc + self._walk(c, body, G, eps)
        return acc

    def compute_accelerations(self, bodies, G, epsilon):
        """Evaluate forces and write to each ``body.acceleration`` in place."""
        n = len(bodies)
        if n == 0:
            return
        # bounding cube with small padding
        pos = np.array([b.position for b in bodies])
        lo, hi = pos.min(0), pos.max(0)
        center = 0.5 * (lo + hi)
        half = 0.5 * np.max(hi - lo) * 1.01 + 1e-10
        root = self._make_node(center, half)
        for b in bodies:
            self._insert(root, b)
        self._compute_quadrupole(root)
        for b in bodies:
            b.acceleration = self._walk(root, b, G, epsilon)



#  Particle-Mesh (FFT Poisson solver)
class ParticleMesh:
    """FFT-based Particle-Mesh (PM) Poisson solver.

    Pipeline per force evaluation:

        1. Cloud-in-Cell (CIC) deposit of particle masses onto a cubic grid.
        2. Forward FFT of the density field.
        3. Solve Poisson in k-space with Green's function ``-4πG / k²``.
           The ``k = 0`` mode is zeroed (mean-density background removed).
        4. Inverse FFT to recover the potential ``φ`` on the grid.
        5. Second-order central differences to obtain ``-∇φ``.
        6. CIC interpolation of the force field back to each particle.

    Complexity: O(N + M log M) where ``M = grid_size³``.

    Boundary conditions: the FFT is inherently periodic, so the solver
    computes the potential of the particle distribution *plus all of its
    periodic replicas*.  For an isolated system use a ``box_size``
    comfortably larger than the particle distribution — the default
    auto-fit pads by ``pad`` on each side, which is adequate for
    clusters but *not* for the solar system.  (The rigorous cure is
    zero-pad to 2× in each dimension; omitted here for compactness.)

    Softening: grid discretisation naturally smooths forces below the
    cell size ``h = box_size / grid_size``, so the ``epsilon`` argument
    is ignored — the effective softening length is ≈ h.
    """

    def __init__(self, grid_size: int = 64, box_size: float | None = None,
                 pad: float = 0.5):
        """
        Parameters
        grid_size : int            grid dimension per side (cube)
        box_size  : float or None  side length of the periodic box;
                                   if None, the box is fit to the
                                   particle bounding box each call.
        pad       : float          relative padding when auto-fitting
                                   (0.5 → box is 2× the particle span).
        """
        self.grid_size = int(grid_size)
        self.box_size = box_size
        self.pad = float(pad)

    def _fit_box(self, positions):
        lo = positions.min(0)
        hi = positions.max(0)
        size = float((hi - lo).max())
        size *= (1.0 + 2.0 * self.pad)
        size = max(size, 1e-30)
        center = 0.5 * (lo + hi)
        origin = center - 0.5 * size
        return origin, size

    def compute_accelerations(self, bodies, G, epsilon):
        n = len(bodies)
        if n == 0:
            return

        pos  = np.array([b.position for b in bodies], dtype=np.float64)
        mass = np.array([b.mass     for b in bodies], dtype=np.float64)

        if self.box_size is None:
            origin, L = self._fit_box(pos)
        else:
            L = float(self.box_size)
            origin = -0.5 * L * np.ones(3)

        N = self.grid_size
        h = L / N

        # normalise particle positions into [0, N) grid coordinates
        x = (pos - origin) / h
        x = np.clip(x, 0.0, N - 1.000001)

        i0 = np.floor(x).astype(np.int64)
        f  = x - i0
        i1 = (i0 + 1) % N                     # periodic wrap

        wx0, wy0, wz0 = 1.0 - f[:, 0], 1.0 - f[:, 1], 1.0 - f[:, 2]
        wx1, wy1, wz1 = f[:, 0],       f[:, 1],       f[:, 2]

        # index tuples for the 8 CIC corners
        corners = [
            (i0[:, 0], i0[:, 1], i0[:, 2], wx0 * wy0 * wz0),
            (i1[:, 0], i0[:, 1], i0[:, 2], wx1 * wy0 * wz0),
            (i0[:, 0], i1[:, 1], i0[:, 2], wx0 * wy1 * wz0),
            (i1[:, 0], i1[:, 1], i0[:, 2], wx1 * wy1 * wz0),
            (i0[:, 0], i0[:, 1], i1[:, 2], wx0 * wy0 * wz1),
            (i1[:, 0], i0[:, 1], i1[:, 2], wx1 * wy0 * wz1),
            (i0[:, 0], i1[:, 1], i1[:, 2], wx0 * wy1 * wz1),
            (i1[:, 0], i1[:, 1], i1[:, 2], wx1 * wy1 * wz1),
        ]

        # 1. mass deposit 
        rho = np.zeros((N, N, N), dtype=np.float64)
        for ix, iy, iz, w in corners:
            np.add.at(rho, (ix, iy, iz), mass * w)
        rho /= h ** 3

        # 2–4. FFT Poisson: φ_k = −4πG ρ_k / k² 
        kx = 2.0 * np.pi * np.fft.fftfreq(N,  d=h)
        ky = 2.0 * np.pi * np.fft.fftfreq(N,  d=h)
        kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=h)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX * KX + KY * KY + KZ * KZ
        k2[0, 0, 0] = 1.0                     # avoid div-by-zero at DC

        rho_hat = np.fft.rfftn(rho)
        phi_hat = -4.0 * np.pi * G * rho_hat / k2
        phi_hat[0, 0, 0] = 0.0
        phi = np.fft.irfftn(phi_hat, s=(N, N, N))

        # 5. force field = −∇φ via 2nd-order central differences
        ax_grid = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * h)
        ay_grid = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * h)
        az_grid = -(np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * h)

        # 6. CIC interpolate forces back to particles
        def _interp(field):
            vals = np.zeros(n)
            for ix, iy, iz, w in corners:
                vals += field[ix, iy, iz] * w
            return vals

        ax = _interp(ax_grid)
        ay = _interp(ay_grid)
        az = _interp(az_grid)

        for i, b in enumerate(bodies):
            b.acceleration = np.array([ax[i], ay[i], az[i]], dtype=np.float64)
