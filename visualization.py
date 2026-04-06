"""
Visualization for N-body simulations.

Produces a multi-panel figure:
  • 3D orbit plot (top-left)
  • XY, XZ, YZ projections (top-right, bottom-left, bottom-right)
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
    r = max(xr, yr) * margin * 0.5
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal")


def plot_results(universe, title: str = "N-Body Simulation", save_path: str | None = None):
    """
    Generate a comprehensive results figure.

    Parameters
    universe : Universe   A universe that has already been run.
    title : str           Figure super-title.
    save_path : str       If given, save to this path instead of showing.
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
        half = max(ranges) * 0.6
        cx = (max(all_x)+min(all_x))/2
        cy = (max(all_y)+min(all_y))/2
        cz = (max(all_z)+min(all_z))/2
        ax3d.set_xlim(cx-half, cx+half)
        ax3d.set_ylim(cy-half, cy+half)
        ax3d.set_zlim(cz-half, cz+half)

    ax3d.set_xlabel("X (m)", fontsize=9)
    ax3d.set_ylabel("Y (m)", fontsize=9)
    ax3d.set_zlabel("Z (m)", fontsize=9)
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
        t_days = t / 86400.0  # convert to days for readability

        # energy panel
        ax_e = fig.add_subplot(gs[2, 0], **dark_ax_kw)
        E = np.array(universe.diag_E)
        E0 = E[0] if E[0] != 0 else 1.0
        rel_err = (E - E[0]) / abs(E0)
        ax_e.plot(t_days, rel_err, color="#ff6b6b", lw=1)
        ax_e.set_ylabel("ΔE / |E₀|", color="#ccc", fontsize=9)
        ax_e.set_xlabel("Time (days)", color="#aaa", fontsize=8)
        ax_e.set_title("Relative Energy Error", color="#ccc", fontsize=10)
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
        L_mag = np.linalg.norm(L, axis=1)
        L0 = L_mag[0]
        if L0 > 1e-12:
            rel_L = (L_mag - L0) / L0
            max_l_drift = np.max(np.abs(rel_L))
            drift_label = f"max |ΔL/L₀| = {max_l_drift:.2e}"
            ylabel = "ΔL / |L₀|"
        else:
            rel_L = L_mag - L0
            max_l_drift = np.max(np.abs(rel_L))
            drift_label = f"max |ΔL| = {max_l_drift:.2e}"
            ylabel = "ΔL (absolute)"
        ax_l.plot(t_days, rel_L, color="#4ecdc4", lw=1)
        ax_l.set_ylabel(ylabel, color="#ccc", fontsize=9)
        ax_l.set_xlabel("Time (days)", color="#aaa", fontsize=8)
        ax_l.set_title("Angular Momentum Drift", color="#ccc", fontsize=10)
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
        ax_eb.set_ylabel("Energy (J)", color="#ccc", fontsize=9)
        ax_eb.set_xlabel("Time (days)", color="#aaa", fontsize=8)
        ax_eb.set_title("Energy Breakdown", color="#ccc", fontsize=10)
        ax_eb.tick_params(colors="#888", labelsize=7)
        ax_eb.spines[:].set_color("#333")
        ax_eb.legend(fontsize=7, facecolor="#222", edgecolor="#555", labelcolor="white", loc="best")

    fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=0.97)

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
    L_mag = np.linalg.norm(L, axis=1)
    L0 = L_mag[0]
    if L0 > 1e-12:
        max_dL = np.max(np.abs(L_mag - L0)) / L0
        L_label = f"max |ΔL / L₀|  = {max_dL:.4e}"
    else:
        max_dL_abs = np.max(np.abs(L_mag - L0))
        L_label = f"max |ΔL|       = {max_dL_abs:.4e}  (L₀ ≈ 0)"

    P = np.array(universe.diag_P)
    P_mag = np.linalg.norm(P, axis=1)
    P0 = P_mag[0]
    if P0 > 1e-12:
        max_dP = np.max(np.abs(P_mag - P0)) / P0
        P_label = f"max |ΔP / P₀|  = {max_dP:.4e}"
    else:
        max_dP_abs = np.max(np.abs(P_mag - P0))
        P_label = f"max |ΔP|       = {max_dP_abs:.4e}  (P₀ ≈ 0)"

    t_days = universe.diag_time[-1] / 86400
    print("=" * 60)
    print(f"  Conservation Summary  ({t_days:.1f} days, "
          f"{len(universe.diag_E)} steps)")
    print("=" * 60)
    print(f"  Energy:    max |ΔE / E₀|  = {max_dE:.4e}")
    print(f"  Ang. Mom:  {L_label}")
    print(f"  Lin. Mom:  {P_label}")
    print(f"  E₀ = {E[0]:.6e} J")
    print(f"  E_final = {E[-1]:.6e} J")
    print("=" * 60)