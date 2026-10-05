"""
N-body examples and command-line entry point.

Solar system, figure-eight choreography, and random cluster.
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


# Command-line options

def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="cluster",
                        choices=("solar", "figure-eight", "cluster"))
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
    parser.add_argument("--no-plot", action="store_true", help="run using only NumPy")
    args = parser.parse_args(argv)

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
