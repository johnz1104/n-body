"""
N-body examples and command-line entry point.

Solar system, figure-eight choreography, random cluster, and radial-velocity fit.
"""

import numpy as np

from core import Body
from universe import Universe

COLORS = ("#ff6b6b", "#4ecdc4", "#ffd93d", "#6c5ce7", "#fd79a8", "#74b9ff")


def _present(u, title, save_path, plot):
    if plot:
        from visualization import plot_results, print_conservation_summary
        print_conservation_summary(u)
        plot_results(u, title=title, save_path=save_path)


#  SCENARIO 1 — Solar System

def solar_system(method="leapfrog", force_method="direct",
                 theta=0.5, years=2.0, save_path=None, *,
                 steps=None, plot=True, progress=True, **force_options):
    """
    Inner + outer solar system.
    Default: leapfrog with 1-day timestep for ~2 years.
    """
    dt = 86400.0  # 1 day
    steps = int(years * 365.25) if steps is None else steps

    u = Universe(dt=dt, epsilon=1e6, theta=theta,
                 force_method=force_method, **force_options)

    # Sun
    u.add_body(Body("Sun", 1.989e30, [0, 0, 0], [0, 0, 0], "#FDB813"))

    # [name, mass(kg), semi-major axis(m), orbital speed(m/s), incl(deg), color]
    planets = [
        ("Mercury",  3.285e23,  5.791e10, 47360, 7.00, "#B0B0B0"),
        ("Venus",    4.867e24,  1.082e11, 35020, 3.40, "#E8CDA0"),
        ("Earth",    5.972e24,  1.496e11, 29780, 0.00, "#4A90D9"),
        ("Mars",     6.390e23,  2.279e11, 24070, 1.85, "#C1440E"),
        ("Jupiter",  1.898e27,  7.785e11, 13070, 1.30, "#C88B3A"),
        ("Saturn",   5.683e26,  1.432e12,  9680, 2.50, "#E8D191"),
        ("Uranus",   8.681e25,  2.867e12,  6800, 0.77, "#7EC8E3"),
        ("Neptune",  1.024e26,  4.515e12,  5430, 1.77, "#3D5EAB"),
    ]

    for name, mass, dist, speed, incl_deg, color in planets:
        incl = np.radians(incl_deg)
        pos = [dist, 0.0, 0.0]
        vel = [0.0, speed * np.cos(incl), speed * np.sin(incl)]
        u.add_body(Body(name, mass, pos, vel, color))

    print(f"\n{'─'*60}")
    print(f"  Solar System — {method} / {force_method} / {steps} steps")
    print(f"{'─'*60}")
    u.run(steps, method=method, progress=progress)
    _present(u, f"Solar System ({method}, {force_method})", save_path, plot)
    return u


#  SCENARIO 2 — Figure-8 Three-Body (stability benchmark)
def figure_eight(method="rk4", steps=20000, save_path=None, *,
                 force_method="direct", plot=True, progress=True, **force_options):
    """
    The Chenciner-Montgomery figure-8 solution — a periodic three-body
    choreography.  All three equal masses trace the same figure-8 curve.

    Uses G = 1, m = 1 normalised units.  Period T ≈ 6.3259.
    """
    # initial conditions from Moore / Chenciner-Montgomery
    p = 0.347111
    v = 0.532728

    if not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError("figure_eight steps must be a positive integer")
    dt = 6.3259 / steps * 10  # ~10 periods

    u = Universe(dt=dt, G=1.0, epsilon=1e-8, force_method=force_method, **force_options)

    u.add_body(Body("A", 1.0,
                     [-1.0, 0.0, 0.0],
                     [p, v, 0.0], "#ff6b6b"))
    u.add_body(Body("B", 1.0,
                     [1.0, 0.0, 0.0],
                     [p, v, 0.0], "#4ecdc4"))
    u.add_body(Body("C", 1.0,
                     [0.0, 0.0, 0.0],
                     [-2*p, -2*v, 0.0], "#ffd93d"))

    print(f"\n{'─'*60}")
    print(f"  Figure-8 Three-Body — {method} / {steps} steps")
    print(f"{'─'*60}")
    u.run(steps, method=method, progress=progress)
    _present(u, f"Figure-8 Choreography ({method}, {force_method})", save_path, plot)
    return u


#  SCENARIO 3 — Random Cluster (Barnes-Hut stress test)
def random_cluster(n_bodies=64, method="leapfrog", theta=0.7,
                   steps=500, save_path=None, *, force_method="barnes-hut",
                   plot=True, progress=True, **force_options):
    """
    Random spherical cluster of equal-mass bodies.
    Select any gravity backend; use an explicit box_size for periodic PM.
    """
    if not isinstance(n_bodies, (int, np.integer)) or n_bodies <= 0:
        raise ValueError("n_bodies must be a positive integer")
    rng = np.random.default_rng(42)
    mass = 1e26  # each body

    u = Universe(dt=3600.0, G=6.6743e-11, epsilon=1e7,
                 theta=theta, force_method=force_method, **force_options)

    R = 1e10  # cluster radius

    for i in range(n_bodies):
        # uniform in sphere via rejection
        while True:
            pos = rng.uniform(-R, R, 3)
            if np.linalg.norm(pos) <= R:
                break
        vel = rng.normal(0, 500, 3)  # mild velocity dispersion
        c = COLORS[i % len(COLORS)]
        u.add_body(Body(f"m{i}", mass, pos, vel, c))

    print(f"\n{'─'*60}")
    print(f"  Random Cluster — {n_bodies} bodies, θ={theta}, {method}")
    print(f"{'─'*60}")
    u.run(steps, method=method, progress=progress)
    _present(u, f"Random Cluster (N={n_bodies}, {force_method}, {method})", save_path, plot)
    return u

def compare_force_methods(save_path=None):
    """Compare isolated force solvers on identical solar initial conditions."""
    from visualization import plot_force_comparison
    runs = []
    for force, theta in [("direct", 0), ("barnes-hut", 0.5), ("multipole", 0.5)]:
        u = solar_system(force_method=force, theta=theta, years=1,
                         plot=False, progress=False)
        label = "direct" if force == "direct" else f"{force} θ={theta}"
        runs.append((label, u))
    plot_force_comparison(runs, save_path=save_path)
    return runs


#  SCENARIO 4 — Radial-velocity fit

# AU, Julian years, solar masses
RV_G = 4 * np.pi**2
AU_PER_YEAR_TO_M_PER_S = 149597870700.0 / (365.25 * 86400.0)


def planetary_state(parameters):
    """Star and two planets, with adjustable inner-planet mass and phase.

    Mass is in solar masses and phase in radians. Initial speeds approximate
    circular two-body orbits; the simulation includes all mutual interactions.
    """
    from core import _jax_modules, make_state, barycentric_state
    _, jnp = _jax_modules()
    masses = jnp.array([1.0, parameters["planet_mass"], 0.0005])
    phases = jnp.array([parameters["phase"], 2.2])
    radii = jnp.array([0.5, 1.1])
    c = jnp.cos(phases)
    s = jnp.sin(phases)
    zero = jnp.zeros(2)
    planet_positions = radii[:, None] * jnp.stack((c, s, zero), axis=1)
    speeds = jnp.sqrt(RV_G * (masses[0] + masses[1:]) / radii)
    planet_velocities = speeds[:, None] * jnp.stack((-s, c, zero), axis=1)

    # Add the star, then shift to the center-of-mass frame.
    positions = jnp.concatenate((jnp.zeros((1, 3)), planet_positions))
    velocities = jnp.concatenate((jnp.zeros((1, 3)), planet_velocities))
    state = make_state(masses, positions, velocities)
    return barycentric_state(state)


def radial_velocity_demo(method="leapfrog", steps=1000, *, plot=True,
                         output_dir="results/inference", seed=42):
    """Fit a planet's mass and phase from synthetic stellar radial velocities.

    Generate observations at half the fitting timestep, with 0.3 m/s noise.
    Write the fit summary, data and optional plot to output_dir.
    """
    import json
    from pathlib import Path
    from core import enable_autodiff, _jax_modules
    from universe import simulate
    from observables import radial_velocity, sample_observable
    from inference import fit_parameters

    if (isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or steps < 100):
        raise ValueError("RV demo needs at least 100 steps over its two-year baseline")
    enable_autodiff()
    jax, _ = _jax_modules()
    dt = 2.0 / steps
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 2.0, 96))
    truth = {"planet_mass": 0.001, "phase": 0.7}
    initial = {"planet_mass": 0.0006, "phase": 1.1}

    def model(parameters, refinement=1):
        state = planetary_state(parameters)
        trajectory = simulate(state, dt=dt/refinement,
                              steps=steps*refinement, method=method, G=RV_G, epsilon=0.0)
        rv = radial_velocity(trajectory, line_of_sight=(1, 0, 0)) * AU_PER_YEAR_TO_M_PER_S
        return trajectory, rv

    def predict(parameters):
        trajectory, rv = model(parameters)
        return sample_observable(trajectory, rv, times)

    # synthetic observations at a finer timestep
    truth_trajectory, truth_curve = model(truth, refinement=2)
    noiseless = np.asarray(sample_observable(truth_trajectory, truth_curve, times))
    sigma = np.full(times.shape, 0.3)  # m/s, independent Gaussian noise
    observations = noiseless + rng.normal(size=times.size) * sigma

    # fit the inner planet, keeping geometry and other parameters fixed
    bounds = {"planet_mass": (0.0001, 0.003), "phase": (-np.pi, np.pi)}
    fit = fit_parameters(predict, initial, observations, sigma,
                         bounds=bounds)
    fitted_trajectory, fitted_curve = model(fit.parameters)
    _, initial_curve = model(initial)
    sensitivity = jax.jacrev(predict)(fit.parameters)

    # save numerical results
    summary = {
        "integrator": method,
        "steps": steps,
        "dt_years": dt,
        "truth_dt_years": dt/2,
        "seed": seed,
        "observations": times.size,
        "noise_m_per_s": 0.3,
        "true_parameters": truth,
        "initial_parameters": initial,
        "fitted_parameters": fit.parameters,
        "success": fit.success,
        "message": fit.message,
        "nfev": fit.nfev,
        "initial_chi_squared": 2*fit.initial_cost,
        "chi_squared": fit.chi_squared,
        "degrees_of_freedom": times.size - len(initial),
        "units": {"planet_mass": "solar masses", "phase": "radians"},
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(directory / "observations.npz", times=times, observations=observations,
             uncertainties=sigma, fitted=fit.prediction, noiseless=noiseless,
             model_times=np.asarray(fitted_trajectory.times),
             model_rv=np.asarray(fitted_curve), cost_history=fit.cost_history,
             d_rv_d_mass=np.asarray(sensitivity["planet_mass"]),
             d_rv_d_phase=np.asarray(sensitivity["phase"]))
    if plot:
        from visualization import plot_radial_velocity_fit

        plot_sensitivity = {
            "Mass (fractional change)":
                np.asarray(sensitivity["planet_mass"]) * fit.parameters["planet_mass"],
            "Phase (per radian)": np.asarray(sensitivity["phase"]),
        }
        plot_radial_velocity_fit(times, observations, sigma,
                                 np.asarray(fitted_trajectory.times),
                                 np.asarray(initial_curve), np.asarray(fitted_curve),
                                 fit.prediction, fit.cost_history, plot_sensitivity,
                                 save_path=directory / "fit.png")
    print(json.dumps(summary, indent=2))
    return fit


# Command-line options

def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="cluster",
                        choices=("solar", "figure-eight", "cluster", "rv-fit"))
    parser.add_argument("--integrator", choices=Universe.INTEGRATORS, default="leapfrog")
    parser.add_argument("--force", choices=Universe.FORCE_METHODS, default="direct")
    parser.add_argument("--steps", type=int, help="step count; scenario default if omitted")
    parser.add_argument("--bodies", type=int, default=64, help="cluster particle count")
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--multipole-order", type=int, choices=(0, 2), default=2)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--box-size", type=float, help="periodic box side in scenario position units")
    parser.add_argument("--record-every", type=int, default=1, help="0 disables histories and diagnostics")
    parser.add_argument("--output", help="figure path (default: results/<scenario>_<integrator>_<force>.png)")
    parser.add_argument("--output-dir", default="results/inference", help="RV fit data and figure directory")
    parser.add_argument("--seed", type=int, default=42, help="RV observation/noise seed")
    parser.add_argument("--no-plot", action="store_true", help="disable plotting (rv-fit still needs JAX/SciPy)")
    args = parser.parse_args(argv)

    # inference example
    if args.scenario == "rv-fit":
        if args.force != "direct":
            parser.error("rv-fit requires --force direct")
        if args.output:
            parser.error("rv-fit uses --output-dir for its figure and numerical data")
        try:
            result = radial_velocity_demo(method=args.integrator,
                                           steps=1000 if args.steps is None else args.steps,
                                           plot=not args.no_plot,
                                           output_dir=args.output_dir, seed=args.seed)
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))
        if not result.success:
            parser.exit(1, "Fit did not converge; inspect fit.json for optimizer diagnostics.\n")
        return result

    # forward simulation examples
    if args.record_every == 0 and not args.no_plot:
        parser.error("--record-every 0 requires --no-plot")
    path = None
    if not args.no_plot:
        path = Path(args.output or f"results/{args.scenario}_{args.integrator}_{args.force}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
    options = dict(method=args.integrator, force_method=args.force, theta=args.theta,
                   multipole_order=args.multipole_order, grid_size=args.grid_size,
                   box_size=args.box_size, record_every=args.record_every,
                   plot=not args.no_plot, save_path=path)
    if args.steps is not None:
        options["steps"] = args.steps
    try:
        if args.scenario == "solar":
            return solar_system(**options)
        if args.scenario == "figure-eight":
            return figure_eight(**options)
        return random_cluster(n_bodies=args.bodies, **options)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
