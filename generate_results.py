"""Run example simulations and delegate all rendering to visualization.py.

Existing results/ images are historical; regenerating intentionally replaces them.
Use --output-dir to keep a new run separate.
"""

from pathlib import Path
import numpy as np
from core import Body
from universe import Universe
from main import solar_system, figure_eight, random_cluster, compare_force_methods

OUT = Path(__file__).parent / "results"


def gen_solar_system():
    return solar_system(save_path=OUT / "solar_system.png")


def gen_figure_eight():
    return figure_eight(save_path=OUT / "figure_eight.png")


def gen_random_cluster():
    return random_cluster(save_path=OUT / "random_cluster.png")


def gen_force_comparison():
    return compare_force_methods(save_path=OUT / "force_comparison.png")


def gen_fmm_solar():
    """Historical output filename retained; this is a quadrupole tree, not FMM."""
    return solar_system(force_method="multipole", theta=0.3,
                        save_path=OUT / "fmm_solar.png")


def gen_pm_cluster():
    return random_cluster(force_method="particle-mesh", grid_size=32,
                          box_size=6e10, steps=300, save_path=OUT / "pm_cluster.png")


def validation_runs():
    """Measured circular, eccentric and figure-eight direct-gravity baselines."""
    runs = []
    for eccentricity in (0, 0.5):
        for integrator in Universe.INTEGRATORS:
            # Total mass and semimajor separation are one: period = 2 pi.
            u = Universe(dt=2*np.pi/1000, G=1, epsilon=0)
            separation = 1-eccentricity
            speed = np.sqrt((1+eccentricity)/(1-eccentricity))
            u.add_body(Body("a", 0.5, [-separation/2, 0, 0], [0, -speed/2, 0]))
            u.add_body(Body("b", 0.5, [separation/2, 0, 0], [0, speed/2, 0]))
            u.run(5000, method=integrator, progress=False)
            runs.append((f"Kepler e={eccentricity}", integrator, u))
    for integrator in Universe.INTEGRATORS:
        u = figure_eight(method=integrator, steps=10000, plot=False, progress=False)
        runs.append(("Figure-8", integrator, u))
    return runs


def gen_validation_table():
    from visualization import plot_validation_table
    plot_validation_table(validation_runs(), OUT / "validation_table.png")


def gen_figure_eight_gif():
    from visualization import animate_trajectories
    u = Universe(dt=6.3259/600, G=1, epsilon=1e-8)
    p, v = 0.347111, 0.532728
    u.add_body(Body("A", 1, [-1, 0, 0], [p, v, 0], "#ff6b6b"))
    u.add_body(Body("B", 1, [1, 0, 0], [p, v, 0], "#4ecdc4"))
    u.add_body(Body("C", 1, [0, 0, 0], [-2*p, -2*v, 0], "#ffd93d"))
    u.run(1200, method="rk4", progress=False)
    animate_trajectories(u, OUT / "figure_eight.gif")


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    for generate in (gen_solar_system, gen_figure_eight, gen_random_cluster,
                     gen_force_comparison, gen_fmm_solar, gen_pm_cluster,
                     gen_validation_table, gen_figure_eight_gif):
        generate()
    print(f"All eight assets written to {OUT}")
