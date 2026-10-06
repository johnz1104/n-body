# nbody

A gravitational N-body toolkit with **differentiable simulations and parameter
fitting**. It explores RK4 and leapfrog integration, direct and multipole tree
forces, and periodic particle-mesh gravity. Its JAX path computes derivatives
through direct-gravity simulations to fit masses and initial conditions to
astronomical observations.

`core.py` owns particles and gravity primitives. `universe.py` drives the
simulation, integration, and diagnostics. `visualization.py` consumes recorded
results without running the physics. Differentiable RK4 and leapfrog currently
support **direct gravity only**; tree and particle-mesh methods remain NumPy
forward solvers.

## Quick start

Python 3.10+ (tested with Python 3.12). Only NumPy is required for headless forward
simulations. Matplotlib produces plots and Pillow produces GIFs. JAX and SciPy
are optional dependencies for differentiable simulation and fitting.

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

To install the differentiable extension and run its example:

```bash
python -m pip install -r requirements-diff.txt
python main.py --scenario rv-fit --integrator leapfrog
# Also supports --integrator rk4 and --no-plot
# Save a separate run without replacing a previous example:
python main.py --scenario rv-fit --integrator rk4 --output-dir /tmp/nbody-rv-rk4
```

The demo generates 96 noisy stellar radial-velocity measurements for a star and
two planets, then fits the inner planet's mass and initial orbital phase.
Observations are generated at twice the fit's integration resolution, with a
fixed random seed. Star mass, viewing geometry, other orbital parameters, and
the outer planet are fixed. Units are AU, Julian years, and solar masses, with
radial velocities converted to m/s.

Outputs in `results/inference/` (ignored by Git):

- `fit.json`: true/initial/fitted parameters, settings, optimizer status, and χ².
- `observations.npz`: observations, uncertainties, fitted signal, sensitivities,
  and optimization trial costs.
- `fit.png`: signal, residuals, sensitivity curves, and optimization progress
  (unless `--no-plot`).

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

## Differentiable simulation

Call `enable_autodiff()` **before creating JAX arrays**. This explicitly enables
JAX's process-wide float64 setting; the differentiable API rejects float32 mode.
Plain imports of `core`, `universe`, `observables`, and `inference` do not load
JAX, SciPy, or plotting libraries.

```python
from core import enable_autodiff, make_state
enable_autodiff()

import jax
from universe import simulate
from observables import radial_velocity

state = make_state(
    masses=[0.5, 0.5],
    positions=[[-0.5, 0, 0], [0.5, 0, 0]],
    velocities=[[0, -0.5, 0], [0, 0.5, 0]],
)

trajectory = simulate(state, dt=0.01, steps=200, method="leapfrog", G=1.0)
# times: (201,), positions and velocities: (201, 2, 3), including initial state

def final_velocity(s):
    track = simulate(s, dt=0.01, steps=200, method="leapfrog", G=1.0)
    return radial_velocity(track, body_index=0, line_of_sight=(1, 0, 0))[-1]

gradient = jax.jit(jax.grad(final_velocity))(state)
# gradient.masses: (2,), gradient.positions/velocities: (2, 3)
```

`ParticleState` and `Trajectory` are immutable named tuples of JAX arrays and
work with `grad`, `jacrev`, `jit`, and `vmap`. `make_state` validates shapes,
finite values, and positive masses for concrete inputs; when tracing under
`jit`/`grad`, values cannot be checked in Python. Constrain optimized parameters
to valid values with bounds or transformations such as log mass.

For an existing direct-gravity `Universe`, use:

```python
state = u.differentiable_state()
trajectory = u.simulate_differentiable(200, method="rk4", state=state)
```

Here `u` must have `force_method="direct"` and at least one body; the tree and
periodic examples above intentionally do not support this operation.

This snapshots the universe's current state and uses its `dt`, `G`, `epsilon`,
and current time. It does **not** advance `u`, change its bodies, or append
histories. Pass `state` explicitly to differentiate its inputs. Pure
`universe.simulate()` is the main array API; its configuration keywords are
static. When compiling a closure over a Universe, keep its configuration fixed
or recreate the compiled function after changes.

`core.barycentric_state` centers initial positions and velocities on the center
of mass while preserving gradients. `core.differentiable_energy` computes the
same isolated Plummer energy as the NumPy driver.

## Observations and fitting

`observables.py` provides:

- `radial_velocity`: projected velocity, positive away from the observer; the
  normalized line of sight points from observer to system. An additive systemic
  velocity can itself be a fit parameter.
- `sky_plane_positions`: two coordinates in an orthonormal sky basis, optionally
  relative to another body. A supplied distance converts length offsets into
  small-angle offsets in radians. Observer geometry is fixed.
- `sample_observable`: linear interpolation at observation times. Eager calls
  reject out-of-range times; compiled calls return NaNs outside the simulation
  interval, which fitting rejects. Interpolation error must be checked along
  with timestep error.

The fitting interface accepts any pure JAX-compatible prediction function and
an explicit dictionary of scalar parameters to vary. For example:

```python
import jax.numpy as jnp
from inference import fit_parameters
from observables import radial_velocity, sample_observable
from main import planetary_state, RV_G, AU_PER_YEAR_TO_M_PER_S

times = jnp.linspace(0, 2, 80)

def predict(parameters):
    track = simulate(planetary_state(parameters), dt=0.002, steps=1000,
                     method="rk4", G=RV_G)
    rv = radial_velocity(track, line_of_sight=(1, 0, 0)) * AU_PER_YEAR_TO_M_PER_S
    return sample_observable(track, rv, times)

# Replace this noiseless illustration with observed velocities in m/s.
observed = predict({"planet_mass": 0.001, "phase": 0.7})
fit = fit_parameters(
    predict, {"planet_mass": 0.0006, "phase": 1.1}, observed,
    uncertainties=0.3,
    bounds={"planet_mass": (0.0001, 0.003), "phase": (-jnp.pi, jnp.pi)},
)
print(fit.success, fit.parameters, fit.chi_squared)
```

The optimizer uses SciPy bounded least squares with a JAX Jacobian. Its objective
is `0.5 * sum(((prediction - observation) / uncertainty)**2)`, assuming independent
Gaussian measurement errors. Predictions must match the observation shape;
uncertainties may be a positive scalar or a broadcastable array. Fixed quantities
stay inside the prediction function; positions and velocities can be fitted by
mapping named scalar parameters into the initial state in that function.

`FitResult` includes parameter values, predictions, normalized residuals, the
weighted residual Jacobian, cost, trial cost history, optimizer status, and
evaluation count. `success` reports optimizer convergence, not a unique or
correct physical solution. Posterior sampling and uncertainty estimates are
outside this first release. Initial guesses, parameter degeneracies, and
correlated noise require additional analysis for real observations.

### Numerical scope

- Differentiation follows the **discrete numerical simulation**. Validate both
  trajectories and derivatives under timestep refinement.
- This is a fixed-particle, fixed-step Newtonian point-mass model with optional
  Plummer softening. Exact unsoftened collisions are singular; mergers and
  event-time derivatives are unsupported.
- Full trajectories and reverse-mode intermediates consume memory. Direct
  forces use O(N²) work/storage per force evaluation; this initial release is
  intended for small systems, not large cosmological simulations.
- Long chaotic trajectories can have very large, difficult-to-use derivatives.
- The current tree decisions and PM grid operations are **not** differentiated.
  Those backends explicitly reject `simulate_differentiable` rather than silently
  returning incomplete gradients.
- RV and sky projections omit light-travel time, relativity, stellar activity,
  transit light curves, and instrument effects.

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

Differentiable tests additionally cover agreement with the independent NumPy
solver, mass/position/velocity gradients against finite differences, derivative
convergence toward an analytic circular binary, energy-gradient consistency,
momentum and energy behavior, single-particle self derivatives, JIT/batching,
observation geometry and interpolation, bounded weighted fitting, and parameter
recovery from noisy observations generated at finer resolution. With JAX absent,
the differentiable test module skips; `requirements-dev.txt` installs it so the
full suite runs.

The existing images and GIF under `results/` are **historical showcase assets**
from the earlier implementation. Their embedded numbers are not current
validation evidence. `generate_results.py` now computes validation table entries
from actual runs rather than hard-coded values. It delegates every plot and
animation to `visualization.py`. Its legacy `fmm_solar.png` filename is retained
for compatibility but the solver is the quadrupole tree described above.

## Project layout

```text
core.py                 Particle states, NumPy/JAX forces, tree and mesh solvers
universe.py             Simulation drivers, NumPy/JAX RK4/leapfrog, diagnostics
observables.py          Radial velocities, sky projection, observation sampling
inference.py            Bounded fitting with autodiff Jacobians
visualization.py        Trajectories, inference plots, tables, animations
main.py                 Forward scenarios, RV inference demo, CLI
generate_results.py     Showcase simulation recipes
stub.py                 Compatibility imports and attach(), no solver duplication
tests/test_solver.py    Numerical and API regressions
tests/test_differentiable.py  Derivative and inference regressions
requirements*.txt       Runtime and development dependencies
results/                Historical gallery
```

Existing `from stub import FastMultipole, ParticleMesh, attach` usage still
works for the built-in backends. `FastMultipole` is a legacy name for
`MultipoleExpansion`; `attach` now configures native dispatch without replacing
`Universe` methods. Prefer the constructor API for new work.

## Possible extensions

Differentiable particle-mesh and multipole backends, memory-efficient adjoints,
more complete observation models, and posterior sampling are future work. This
project makes no first-ever claim; related research software includes
[NbodyGradient](https://ericagol.github.io/NbodyGradient.jl/dev/) and
[JaxPM](https://github.com/DifferentiableUniverseInitiative/JaxPM).
