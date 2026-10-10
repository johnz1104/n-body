"""
Visualization for N-body simulations.

Produces a multi-panel figure:
  • 3D orbit plot (top-left)
  • XY and XZ projections (right)
  • Conservation diagnostics strip along the bottom
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


def _fade_cmap(hex_color: str):
    """Create a colormap that fades from transparent to the given color."""
    import matplotlib.colors as mc
    rgb = mc.to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        "fade", [(rgb[0], rgb[1], rgb[2], 0.05),
                 (rgb[0], rgb[1], rgb[2], 0.85)], N=256)


def _plot_trajectory_2d(ax, x, y, color, name, lw=0.8):
    """Draw a trajectory with alpha gradient (recent points brighter)."""
    n = len(x)
    if n < 2:
        return
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(0.08, 0.9, n - 1)
    import matplotlib.colors as mc
    rgb = mc.to_rgb(color)
    colors = [(rgb[0], rgb[1], rgb[2], a) for a in alphas]
    lc = LineCollection(segments, colors=colors, linewidths=lw)
    ax.add_collection(lc)
    ax.plot(x[-1], y[-1], 'o', color=color, ms=5, zorder=5)


def _equal_aspect(ax, xs, ys, margin=1.15):
    """Set equal aspect ratio centered on data."""
    xr = max(xs) - min(xs) if xs else 1
    yr = max(ys) - min(ys) if ys else 1
    r = max(xr, yr, 1e-12) * margin * 0.5
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal")


def plot_results(universe, title: str = "N-Body Simulation", save_path: str | None = None,
                 *, time_scale=1.0, time_unit="simulation units", length_unit="simulation units"):
    """
    Generate a comprehensive results figure.

    Parameters
    universe : Universe   A universe that has already been run.
    title : str           Figure super-title.
    save_path : str       If given, save to this path instead of showing.
    time_scale : float    Divide recorded times by this value (86400 for days).
    """
    bodies = universe.bodies

    # set up figure layout
    fig = plt.figure(figsize=(18, 14), facecolor="#0e0e12")
    gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)

    dark_ax_kw = dict(facecolor="#14141a")

    # 3D orbit 
    ax3d = fig.add_subplot(gs[0:2, 0:2], projection="3d",
                           computed_zorder=False)
    ax3d.set_facecolor("#14141a")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333")
    ax3d.yaxis.pane.set_edgecolor("#333")
    ax3d.zaxis.pane.set_edgecolor("#333")
    ax3d.tick_params(colors="#888", labelsize=7)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.label.set_color("#aaa")

    all_x, all_y, all_z = [], [], []
    for b in bodies:
        h = np.asarray(b.history_pos)
        if len(h) == 0:
            continue
        n = len(h)
        alphas = np.linspace(0.05, 0.85, n)
        # plot trajectory with scatter for alpha control
        ax3d.plot(h[:, 0], h[:, 1], h[:, 2],
                  color=b.color, alpha=0.4, lw=0.7)
        ax3d.scatter([h[-1, 0]], [h[-1, 1]], [h[-1, 2]],
                     color=b.color, s=40, edgecolors="white",
                     linewidths=0.4, zorder=10, label=b.name)
        all_x.extend(h[:, 0])
        all_y.extend(h[:, 1])
        all_z.extend(h[:, 2])

    # equal-aspect 3D
    if all_x:
        ranges = [max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)]
        half = max(*ranges, 1e-12) * 0.6
        cx = (max(all_x)+min(all_x))/2
        cy = (max(all_y)+min(all_y))/2
        cz = (max(all_z)+min(all_z))/2
        ax3d.set_xlim(cx-half, cx+half)
        ax3d.set_ylim(cy-half, cy+half)
        ax3d.set_zlim(cz-half, cz+half)

    ax3d.set_xlabel(f"X ({length_unit})", fontsize=9)
    ax3d.set_ylabel(f"Y ({length_unit})", fontsize=9)
    ax3d.set_zlabel(f"Z ({length_unit})", fontsize=9)
    ax3d.view_init(elev=25, azim=135)
    ax3d.legend(fontsize=7, loc="upper left", framealpha=0.5,
                facecolor="#222", edgecolor="#555", labelcolor="white")

    # 2D projections
    proj_axes = {
        "XY": fig.add_subplot(gs[0, 2], **dark_ax_kw),
        "XZ": fig.add_subplot(gs[1, 2], **dark_ax_kw),
    }
    for label, ax in proj_axes.items():
        ax.set_title(f"{label} Projection", color="#ccc", fontsize=10)
        ax.tick_params(colors="#888", labelsize=7)
        ax.spines[:].set_color("#333")

    for b in bodies:
        h = np.asarray(b.history_pos)
        if len(h) == 0:
            continue
        _plot_trajectory_2d(proj_axes["XY"], h[:, 0], h[:, 1], b.color, b.name)
        _plot_trajectory_2d(proj_axes["XZ"], h[:, 0], h[:, 2], b.color, b.name)

    for b in bodies:
        h = np.asarray(b.history_pos)
        if len(h) == 0:
            continue
        all_x.extend(h[:, 0]); all_y.extend(h[:, 1]); all_z.extend(h[:, 2])

    if all_x:
        _equal_aspect(proj_axes["XY"], all_x, all_y)
        _equal_aspect(proj_axes["XZ"], all_x, all_z)

    proj_axes["XY"].set_xlabel("X", color="#aaa", fontsize=8)
    proj_axes["XY"].set_ylabel("Y", color="#aaa", fontsize=8)
    proj_axes["XZ"].set_xlabel("X", color="#aaa", fontsize=8)
    proj_axes["XZ"].set_ylabel("Z", color="#aaa", fontsize=8)

    # conservation diagnostics
    t = np.array(universe.diag_time)
    if len(t) > 1:
        t_days = t / time_scale

        # energy panel
        ax_e = fig.add_subplot(gs[2, 0], **dark_ax_kw)
        E = np.array(universe.diag_E)
        E0 = E[0] if E[0] != 0 else 1.0
        rel_err = (E - E[0]) / abs(E0)
        ax_e.plot(t_days, rel_err, color="#ff6b6b", lw=1)
        ax_e.set_ylabel("ΔE / |E₀|", color="#ccc", fontsize=9)
        ax_e.set_xlabel(f"Time ({time_unit})", color="#aaa", fontsize=8)
        ax_e.set_title("Mesh Energy Change" if universe.force_method == "particle-mesh"
                       else "Relative Energy Error", color="#ccc", fontsize=10)
        ax_e.tick_params(colors="#888", labelsize=7)
        ax_e.spines[:].set_color("#333")
        ax_e.axhline(0, color="#555", lw=0.5, ls="--")
        # annotate max drift
        max_drift = np.max(np.abs(rel_err))
        ax_e.text(0.98, 0.95, f"max |ΔE/E₀| = {max_drift:.2e}",
                  transform=ax_e.transAxes, ha="right", va="top",
                  fontsize=8, color="#ff6b6b",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a24", ec="#ff6b6b", alpha=0.7))

        # angular momentum panel
        ax_l = fig.add_subplot(gs[2, 1], **dark_ax_kw)
        L = np.array(universe.diag_L)
        L0 = np.linalg.norm(L[0])
        L_change = np.linalg.norm(L - L[0], axis=1)
        if L0 > 1e-12:
            rel_L = L_change / L0
            max_l_drift = np.max(np.abs(rel_L))
            drift_label = f"max |ΔL/L₀| = {max_l_drift:.2e}"
            ylabel = "ΔL / |L₀|"
        else:
            rel_L = L_change
            max_l_drift = np.max(np.abs(rel_L))
            drift_label = f"max |ΔL| = {max_l_drift:.2e}"
            ylabel = "ΔL (absolute)"
        ax_l.plot(t_days, rel_L, color="#4ecdc4", lw=1)
        ax_l.set_ylabel(ylabel, color="#ccc", fontsize=9)
        ax_l.set_xlabel(f"Time ({time_unit})", color="#aaa", fontsize=8)
        ax_l.set_title("Angular Momentum Change" if universe.force_method == "particle-mesh"
                       else "Angular Momentum Drift", color="#ccc", fontsize=10)
        ax_l.tick_params(colors="#888", labelsize=7)
        ax_l.spines[:].set_color("#333")
        ax_l.axhline(0, color="#555", lw=0.5, ls="--")
        ax_l.text(0.98, 0.95, drift_label,
                  transform=ax_l.transAxes, ha="right", va="top",
                  fontsize=8, color="#4ecdc4",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a24", ec="#4ecdc4", alpha=0.7))

        # energy breakdown panel
        ax_eb = fig.add_subplot(gs[2, 2], **dark_ax_kw)
        KE = np.array(universe.diag_KE)
        PE = np.array(universe.diag_PE)
        ax_eb.plot(t_days, KE, color="#ffd93d", lw=0.8, label="KE")
        ax_eb.plot(t_days, PE, color="#6c5ce7", lw=0.8, label="PE")
        ax_eb.plot(t_days, E, color="#ff6b6b", lw=1.0, label="Total")
        ax_eb.set_ylabel("Energy (simulation units)", color="#ccc", fontsize=9)
        ax_eb.set_xlabel(f"Time ({time_unit})", color="#aaa", fontsize=8)
        ax_eb.set_title("Energy Breakdown", color="#ccc", fontsize=10)
        ax_eb.tick_params(colors="#888", labelsize=7)
        ax_eb.spines[:].set_color("#333")
        ax_eb.legend(fontsize=7, facecolor="#222", edgecolor="#555", labelcolor="white", loc="best")

    fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=0.97)
    if universe.force_method == "particle-mesh":
        fig.text(0.5, 0.015, "Periodic mesh: energy includes grid self-energy; angular momentum is not a conserved periodic-box quantity.",
                 ha="center", color="#aaa", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def print_conservation_summary(universe):
    """Print a text summary of conservation quality."""
    if len(universe.diag_E) < 2:
        print("No diagnostic data recorded.")
        return

    E = np.array(universe.diag_E)
    E0 = E[0] if E[0] != 0 else 1.0
    max_dE = np.max(np.abs(E - E[0])) / abs(E0)

    L = np.array(universe.diag_L)
    L_change = np.linalg.norm(L - L[0], axis=1)
    L0 = np.linalg.norm(L[0])
    if L0 > 1e-12:
        max_dL = np.max(L_change) / L0
        L_label = f"max |ΔL / L₀|  = {max_dL:.4e}"
    else:
        max_dL_abs = np.max(L_change)
        L_label = f"max |ΔL|       = {max_dL_abs:.4e}  (L₀ ≈ 0)"

    P = np.array(universe.diag_P)
    P_change = np.linalg.norm(P - P[0], axis=1)
    P0 = np.linalg.norm(P[0])
    if P0 > 1e-12:
        max_dP = np.max(P_change) / P0
        P_label = f"max |ΔP / P₀|  = {max_dP:.4e}"
    else:
        max_dP_abs = np.max(P_change)
        P_label = f"max |ΔP|       = {max_dP_abs:.4e}  (P₀ ≈ 0)"

    print("=" * 60)
    print(f"  Diagnostics  (t={universe.diag_time[-1]:.6g}, "
          f"{universe.steps} steps, {len(universe.diag_E)} samples)")
    print("=" * 60)
    print(f"  Model: {universe.energy_description}")
    if universe.force_method == "particle-mesh":
        print("  Angular momentum is not conserved in a periodic box.")
    print(f"  Energy:    max |ΔE / E₀|  = {max_dE:.4e}")
    print(f"  Ang. Mom:  {L_label}")
    print(f"  Lin. Mom:  {P_label}")
    print(f"  E₀ = {E[0]:.6e}")
    print(f"  E_final = {E[-1]:.6e}")
    print("=" * 60)


def plot_force_comparison(runs, save_path=None):
    """Plot diagnostic drift from already-computed isolated simulations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0e0e12")
    for ax in axes:
        ax.set_facecolor("#14141a")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")
        ax.set_xlabel("Time (days)", color="#aaa")
    for label, universe in runs:
        times = np.array(universe.diag_time) / 86400
        energy = np.array(universe.diag_E)
        angular = np.array(universe.diag_L)
        axes[0].plot(times, (energy-energy[0])/(abs(energy[0]) or 1), label=label)
        axes[1].plot(times, np.linalg.norm(angular-angular[0], axis=1) /
                     (np.linalg.norm(angular[0]) or 1), label=label)
    for ax, title, ylabel in zip(axes, ("Energy Drift", "Angular Momentum Drift"),
                                ("ΔE / |E₀|", "|ΔL| / |L₀|")):
        ax.set_title(title, color="#ccc")
        ax.set_ylabel(ylabel, color="#ccc")
        ax.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white")
    fig.suptitle("Isolated Force Solvers (same initial conditions)", color="white", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_validation_table(runs, save_path):
    """Calculate table entries from recorded runs, including the initial state."""
    rows = []
    for label, integrator, universe in runs:
        energy = np.array(universe.diag_E)
        angular = np.array(universe.diag_L)
        de = np.max(abs(energy-energy[0])) / (abs(energy[0]) or 1)
        dl = np.max(np.linalg.norm(angular-angular[0], axis=1))
        l0 = np.linalg.norm(angular[0])
        angular_text = f"{dl/l0:.2e}" if l0 > 1e-12 else f"{dl:.2e} (absolute)"
        rows.append([label, integrator, f"{de:.2e}", angular_text])
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="#0e0e12")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Test Case", "Integrator", "max |ΔE/E₀|", "max |ΔL|/|L₀|"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#444")
        cell.set_facecolor("#1a1a2e" if r == 0 else "#14141a")
        cell.set_text_props(color="#4ecdc4" if r == 0 else "#ddd")
    fig.suptitle("Measured Orbit Validation — Direct Gravity", color="white", fontsize=14)
    fig.text(0.5, 0.03, "Computed from these runs; near-zero initial angular momentum uses absolute change.",
             ha="center", color="#aaa", fontsize=9)
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def animate_trajectories(universe, save_path, *, stride=2, trail_length=300, fps=30,
                         title="N-Body Trajectories", body_indices=None,
                         length_scale=1.0, length_unit="simulation units",
                         time_scale=1.0, time_unit="simulation units"):
    """Save an XY animation of recorded histories; no simulation runs here.

    body_indices selects which bodies to display without changing the simulation.
    Recorded lengths and times are divided by the supplied display scales.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.colors as mc
    bodies = (universe.bodies if body_indices is None else
              [universe.bodies[i] for i in body_indices])
    histories = [np.asarray(b.history_pos) / length_scale for b in bodies]
    if not histories or not len(histories[0]):
        raise ValueError("Animation requires recorded body histories")
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#0e0e12",
                           layout="constrained")
    ax.set_facecolor("#14141a")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#333")
    positions = np.concatenate(histories)
    _equal_aspect(ax, positions[:, 0].tolist(), positions[:, 1].tolist())
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel(f"X ({length_unit})", color="#aaa")
    ax.set_ylabel(f"Y ({length_unit})", color="#aaa")
    trails, dots = [], []
    for body in bodies:
        trail = LineCollection([], linewidths=2 if len(bodies) <= 10 else 0.8)
        ax.add_collection(trail)
        trails.append(trail)
        dot, = ax.plot([], [], "o", color=body.color,
                       ms=8 if len(bodies) <= 10 else 4, label=body.name)
        dots.append(dot)
    if len(bodies) <= 10:
        ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white",
                  loc="upper right", fontsize=8)
    label = ax.text(0.02, 0.02, "", transform=ax.transAxes, color="#aaa")

    def update(frame):
        for body, history, trail, dot in zip(bodies, histories, trails, dots):
            points = history[max(0, frame-trail_length):frame+1, :2]
            segments = np.stack((points[:-1], points[1:]), axis=1)
            trail.set_segments(segments)
            rgb = mc.to_rgb(body.color)
            trail.set_color([(*rgb, alpha) for alpha in np.linspace(0.05, 0.9, len(segments))])
            dot.set_data([history[frame, 0]], [history[frame, 1]])
        label.set_text(f"t = {universe.diag_time[frame] / time_scale:.2f} {time_unit}")
        return trails + dots + [label]

    frames = list(range(0, len(histories[0]), stride))
    if frames[-1] != len(histories[0]) - 1:
        frames.append(len(histories[0]) - 1)
    animation = FuncAnimation(fig, update, frames=frames,
                              interval=1000/fps, blit=True)
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


# Radial-velocity fit

def plot_radial_velocity_fit(times, observations, uncertainties, model_times,
                             initial_curve, fitted_curve, fitted_observations,
                             cost_history, sensitivities, *, save_path=None):
    """Plot the RV signal, sensitivities, residuals and fitting progress."""
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        signal, sensitivity_ax, residual_ax, cost_ax = axes.ravel()

        # observations and model curves
        signal.errorbar(times, observations, yerr=uncertainties, fmt=".",
                        color="#334155", alpha=0.7, label="Synthetic observations")
        signal.plot(model_times, initial_curve, color="#94a3b8", ls="--", label="Initial guess")
        signal.plot(model_times, fitted_curve, color="#2563eb", label="Fitted N-body model")
        signal.set(title="Stellar radial velocity", xlabel="Time (years)", ylabel="Velocity (m/s)")
        signal.legend(fontsize=8)

        # response to each fitted parameter
        for label, derivative in sensitivities.items():
            sensitivity_ax.plot(times, derivative, label=label)
        sensitivity_ax.set(title="Local sensitivity at fitted parameters", xlabel="Time (years)",
                           ylabel="RV response (m/s per stated change)")
        sensitivity_ax.legend(fontsize=8)

        # residuals in units of the measurement uncertainty
        residuals = (np.asarray(observations) - fitted_observations) / uncertainties
        residual_ax.axhline(0, color="#94a3b8", lw=1)
        residual_ax.plot(times, residuals, ".", color="#2563eb")
        residual_ax.set(title="Observation − fitted model", xlabel="Time (years)",
                        ylabel="Residual / measurement uncertainty")
        # optimizer trial history
        evaluations = np.arange(1, len(cost_history)+1)
        cost_ax.semilogy(evaluations, np.maximum(cost_history, 1e-30),
                        "o-", color="#0f766e", ms=4)
        cost_ax.set(title="Optimization trial evaluations", xlabel="Model/Jacobian evaluation",
                    ylabel="Weighted least-squares cost (χ² / 2)")
        for ax in axes.ravel():
            ax.grid(alpha=0.15)
        fig.suptitle("Differentiable N-body inference · star and two planets", fontsize=15)
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path, dpi=160)
            plt.close(fig)
    return fig
