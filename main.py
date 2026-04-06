"""
N-Body Simulation — Main Script

Scenarios:
  1. Solar system (inner planets, 2 Earth-years)
  2. Figure-8 three-body choreography (stability test)
  3. Random cluster with Barnes-Hut

Each scenario runs, prints conservation diagnostics, and
saves a multi-panel results figure.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from core import Body
from universe import Universe
from visualization import plot_results, print_conservation_summary


#  SCENARIO 1 — Solar System

def solar_system(method="leapfrog", force_method="direct",
                 theta=0.5, years=2.0, save_path=None):
    """
    Inner + outer solar system.
    Default: leapfrog with 1-day timestep for ~2 years.
    """
    dt = 86400.0  # 1 day
    steps = int(years * 365.25 * 86400 / dt)

    u = Universe(dt=dt, epsilon=1e6, theta=theta,
                 force_method=force_method)

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
    u.run(steps, method=method)
    print_conservation_summary(u)
    plot_results(u, title=f"Solar System ({method}, {force_method})",
                 save_path=save_path)
    return u


#  SCENARIO 2 — Figure-8 Three-Body (stability benchmark)
def figure_eight(method="rk4", steps=20000, save_path=None):
    """
    The Chenciner-Montgomery figure-8 solution — a periodic three-body
    choreography.  All three equal masses trace the same figure-8 curve.

    Uses G = 1, m = 1 normalised units.  Period T ≈ 6.3259.
    """
    # initial conditions from Moore / Chenciner-Montgomery
    p = 0.347111
    v = 0.532728

    dt = 6.3259 / steps * 10  # ~10 periods

    u = Universe(dt=dt, G=1.0, epsilon=1e-8, force_method="direct")

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
    u.run(steps, method=method)
    print_conservation_summary(u)
    plot_results(u, title=f"Figure-8 Choreography ({method})",
                 save_path=save_path)
    return u


#  SCENARIO 3 — Random Cluster (Barnes-Hut stress test)
def random_cluster(n_bodies=64, method="leapfrog", theta=0.7,
                   steps=500, save_path=None):
    """
    Random spherical cluster of equal-mass bodies.
    Compares direct vs Barnes-Hut force evaluation.
    """
    rng = np.random.default_rng(42)
    mass = 1e26  # each body

    u = Universe(dt=3600.0, G=6.6743e-11, epsilon=1e7,
                 theta=theta, force_method="barnes-hut")

    R = 1e10  # cluster radius
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_bodies))

    for i in range(n_bodies):
        # uniform in sphere via rejection
        while True:
            pos = rng.uniform(-R, R, 3)
            if np.linalg.norm(pos) <= R:
                break
        vel = rng.normal(0, 500, 3)  # mild velocity dispersion
        c = "#{:02x}{:02x}{:02x}".format(
            int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
        u.add_body(Body(f"m{i}", mass, pos, vel, c))

    print(f"\n{'─'*60}")
    print(f"  Random Cluster — {n_bodies} bodies, θ={theta}, {method}")
    print(f"{'─'*60}")
    u.run(steps, method=method)
    print_conservation_summary(u)
    plot_results(u, title=f"Random Cluster (N={n_bodies}, θ={theta}, {method})",
                 save_path=save_path)
    return u

#  SCENARIO 4 — Comparison: Direct vs Barnes-Hut accuracy
def compare_force_methods(save_path=None):
    """
    Run a small system with both direct and Barnes-Hut, compare energy
    drift to quantify the approximation error from θ.
    """
    import matplotlib.pyplot as plt

    configs = [
        ("direct",     0.0),
        ("barnes-hut", 0.3),
        ("barnes-hut", 0.5),
        ("barnes-hut", 0.8),
        ("barnes-hut", 1.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor="#0e0e12")
    for ax in (ax1, ax2):
        ax.set_facecolor("#14141a")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")

    colors_list = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#6c5ce7", "#fd79a8"]

    for idx, (fm, theta) in enumerate(configs):
        u = Universe(dt=86400, G=6.6743e-11, epsilon=1e6,
                     theta=theta, force_method=fm)
        # Sun + inner 4 planets
        u.add_body(Body("Sun", 1.989e30, [0,0,0], [0,0,0]))
        for name, m, d, v, inc, _ in [
            ("Mercury", 3.285e23, 5.791e10, 47360, 7.0, ""),
            ("Venus",   4.867e24, 1.082e11, 35020, 3.4, ""),
            ("Earth",   5.972e24, 1.496e11, 29780, 0.0, ""),
            ("Mars",    6.390e23, 2.279e11, 24070, 1.85,""),
        ]:
            i = np.radians(inc)
            u.add_body(Body(name, m, [d,0,0], [0, v*np.cos(i), v*np.sin(i)]))

        u.run(365, method="leapfrog", progress=False)

        E = np.array(u.diag_E)
        E0 = E[0] if E[0] != 0 else 1.0
        t_days = np.array(u.diag_time) / 86400

        label = f"direct" if fm == "direct" else f"BH θ={theta}"
        ax1.plot(t_days, (E - E[0])/abs(E0), color=colors_list[idx],
                 lw=1.2, label=label)

        L = np.linalg.norm(np.array(u.diag_L), axis=1)
        L0 = L[0] if L[0] != 0 else 1.0
        ax2.plot(t_days, (L - L[0])/abs(L0), color=colors_list[idx],
                 lw=1.2, label=label)

    ax1.set_xlabel("Time (days)", color="#aaa")
    ax1.set_ylabel("ΔE / |E₀|", color="#ccc")
    ax1.set_title("Energy Drift: Direct vs Barnes-Hut", color="#ccc")
    ax1.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")
    ax1.axhline(0, color="#555", ls="--", lw=0.5)

    ax2.set_xlabel("Time (days)", color="#aaa")
    ax2.set_ylabel("ΔL / |L₀|", color="#ccc")
    ax2.set_title("Angular Momentum Drift", color="#ccc")
    ax2.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")
    ax2.axhline(0, color="#555", ls="--", lw=0.5)

    fig.suptitle("Force Method Comparison (1 year, inner planets)",
                 color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out, exist_ok=True)

    # Run all scenarios
    solar_system(method="leapfrog", force_method="direct",
                 years=2.0, save_path=f"{out}/solar_system.png")

    figure_eight(method="rk4", steps=20000,
                 save_path=f"{out}/figure_eight.png")

    random_cluster(n_bodies=64, theta=0.7, steps=500,
                   save_path=f"{out}/random_cluster.png")

    compare_force_methods(save_path=f"{out}/force_comparison.png")

    print("\n✓ All scenarios complete.")