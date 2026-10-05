# nbody

A NumPy gravity lab with independent choices for **time integration** (RK4 or
leapfrog) and **force evaluation** (direct, Barnes–Hut, multipole expansions,
or periodic particle-mesh).

`core.py` owns particles and gravity primitives. `universe.py` drives the
simulation, integration, and diagnostics. `visualization.py` consumes recorded
results without running the physics.

## Quick start

Python 3.10+ (tested with Python 3.12). Only NumPy is required for headless
simulations. Matplotlib produces plots, Pillow produces GIFs, and pytest
runs the numerical regression tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Quadrupole multipole expansion + leapfrog
python main.py --scenario cluster --force multipole --integrator leapfrog --steps 100

# Periodic particle-mesh + RK4, with a fixed box (scenario lengths are in meters)
python main.py --scenario cluster --force particle-mesh --integrator rk4 \
  --grid-size 32 --box-size 6e10 --steps 100

# Direct reference with no plotting dependency
python main.py --scenario solar --force direct --integrator rk4 --steps 100 --no-plot
```

Plots default to `results/<scenario>_<integrator>_<force>.png`. Use `--output`
to choose a path. Run `python main.py --help` for all options.

```python
from core import Body
from universe import Universe

u = Universe(dt=0.01, G=1.0, epsilon=0.01,
             force_method="multipole", theta=0.5, multipole_order=2)
u.add_body(Body("A", 0.5, [-0.5, 0, 0], [0, -0.5, 0], "#ff6b6b"))
u.add_body(Body("B", 0.5, [ 0.5, 0, 0], [0,  0.5, 0], "#4ecdc4"))
u.run(1000, method="leapfrog")  # method="rk4" works with every force backend

from visualization import plot_results, print_conservation_summary
print_conservation_summary(u)
plot_results(u, save_path="orbit.png")
```

For a periodic experiment, construct the universe with:

```python
u = Universe(dt=0.01, G=1.0, force_method="particle-mesh",
             grid_size=32, box_size=8.0, box_origin=[-4, -4, -4])
# Add bodies, then call u.run(..., method="rk4" or "leapfrog").
```

## Methods

RK4 and leapfrog advance positions and velocities in time. Particle-mesh and
multipole expansions calculate gravitational accelerations. Either integrator
can be combined with either force solver, as well as with the direct and
Barnes–Hut reference backends.

| Choice | Role | Implementation |
|---|---|---|
| `rk4` | Fourth-order fixed-step Runge–Kutta | `Universe.step_rk4()` |
| `leapfrog` | Second-order kick–drift–kick | `Universe.step_leapfrog()` |
| `direct` | Isolated pairwise gravity, O(N²) | `Universe` |
| `barnes-hut` | Isolated monopole tree, typically O(N log N) | Shared octree in `core.py` |
| `multipole` | Isolated expansion through quadrupole order, typically O(N log N) | `MultipoleExpansion` in `core.py` |
| `particle-mesh` | Periodic CIC/FFT gravity, O(N + M log M), M = grid_size³ | `ParticleMesh` in `core.py` |

### RK4 and leapfrog

RK4 uses four force stages and refreshes acceleration at the final state (five
force evaluations total). It is not adaptive or symplectic. Leapfrog uses two
force evaluations and is symplectic for the exact conservative pair force.
A suitable fixed timestep generally gives bounded, oscillatory energy error;
it does not guarantee exact energy conservation or stability at arbitrary dt.
Approximate one-sided tree forces and the interpolated mesh force do not carry
the same Hamiltonian guarantees.

### Multipole expansions

The shared octree accumulates mass, center of mass, and central second moment
`S = sum(m s sᵀ)`. The dipole vanishes about the center of mass. With displacement
`r` from the cell COM to the target and `D = r·r + epsilon²`, the quadrupole force is:

```text
a = -G M r / D^(3/2)
    + G (3 S r + 1.5 trace(S) r) / D^(5/2)
    - 7.5 G (rᵀ S r) r / D^(7/2)
```

This expands the same softened kernel as direct gravity, including its trace
term. `multipole_order=0` selects monopoles; `2` includes quadrupoles.
`theta` sets the cell-size/distance acceptance threshold. Smaller values open
more cells; zero recovers direct summation. Cells containing the target always
open to exclude self-force. Bucket leaves and a depth limit handle coincident
particles; unsoftened coincident pairs raise a clear error.

This is a **quadrupole tree code**, not a full linear-time fast multipole method
with multipole-to-local translations. Tree complexity is distribution-dependent;
pathological trees and theta=0 can approach O(N²).

### Particle-mesh

1. Deposit mass with cloud-in-cell (CIC) weights on a cubic grid.
2. Solve Poisson's equation in Fourier space: `phi_k = -4 pi G rho_k / k²`.
3. Set the zero mode to zero, removing the mean-density background.
4. Compute the force with centered grid differences and interpolate with CIC.

The box is **periodic**, including all periodic images. Particle coordinates
wrap modulo the box for force evaluation; stored trajectories remain unwrapped.
The default origin is `[-L/2, -L/2, -L/2]`. If `box_size` is omitted, it is fit
once on the first nonempty evaluation and then held fixed. An explicit box is
recommended for reproducibility. Padding does not create isolated boundaries.

Force resolution is set by `h = box_size/grid_size`; `epsilon` is unused for
PM. It cannot resolve close encounters below the grid scale. The isolated direct
solver is therefore not an identical-physics PM reference. Validate PM with
periodic analytical fields or a refined mesh in the same box.

For context on the distinctions between tree, FMM, periodic PM and hybrid
methods, see the [GADGET-4 simulation documentation](https://wwwmpa.mpa-garching.mpg.de/gadget4/03_simtypes/)
and [code paper](https://arxiv.org/abs/2010.03567).

## Diagnostics and numerical changes

Histories and diagnostics include the initial state, each sampled step, and
each `run()` endpoint. Repeated runs continue from the current state without
adding a duplicate initial sample. All masses must be positive and state
vectors finite and three-dimensional. Use a consistent unit system throughout.

- Direct and tree runs report the exact isolated Plummer pair energy:
  `PE = -sum(i<j) G mi mj / sqrt(rij² + epsilon²)`.
- PM runs report the periodic mesh diagnostic `PE = 0.5 integral(rho phi) dV`.
  This includes mesh self-energy and is not an exact Hamiltonian for the
  centered-difference/CIC force; its change includes spatial discretization error.
- Diagnostics record kinetic, potential and total energy, plus linear and
  angular momentum vectors. Vector changes detect changes in direction as well
  as magnitude. Angular momentum is not a conserved periodic-box quantity.
- One-sided tree approximations need not conserve total momentum exactly.
- Exact isolated energy diagnostics cost O(N²) even when tree forces are used.
  `record_every=10` reduces their frequency; `record_every=0` disables histories
  and diagnostics for performance experiments. Direct `step_*()` calls obey the
  interval, while `run()` also samples its final state.

The original force softened distances with `r + epsilon` but its potential was
not consistent with that force. Both now use the Plummer kernel. Consequently,
old numerical results are not regression targets for the current solver.

## Validation and existing results

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# Generate new figures without overwriting the original gallery:
python generate_results.py --output-dir /tmp/nbody-results
```

Tests cover orbit convergence order, leapfrog conservation, the gradient of the
softened potential, exact tree limits, quadrupole accuracy, all eight
integrator/backend combinations, coincident particles, PM mass conservation,
self-force, periodic crossings, and analytical Fourier modes through the full
particle-to-grid-to-particle pipeline.

The existing images and GIF under `results/` are **historical showcase assets**
from the earlier implementation. Their embedded numbers are not current
validation evidence. `generate_results.py` now computes validation table entries
from actual runs rather than hard-coded values. It delegates every plot and
animation to `visualization.py`. Its legacy `fmm_solar.png` filename is retained
for compatibility but the solver is the quadrupole tree described above.

## Project layout

```text
core.py                 Body, octree, multipole and particle-mesh forces
universe.py             Force dispatch, RK4/leapfrog and diagnostics
visualization.py        Trajectories, diagnostic plots, tables, animations
main.py                 Forward scenarios and CLI
generate_results.py     Showcase simulation recipes
stub.py                 Compatibility imports and attach(), no solver duplication
tests/test_solver.py    Numerical and API regressions
requirements*.txt       Runtime and development dependencies
results/                Historical gallery
```

Existing `from stub import FastMultipole, ParticleMesh, attach` usage still
works for the built-in backends. `FastMultipole` is a legacy name for
`MultipoleExpansion`; `attach` now configures native dispatch without replacing
`Universe` methods. Prefer the constructor API for new work.
