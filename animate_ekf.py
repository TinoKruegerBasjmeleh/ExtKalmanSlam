#!/usr/bin/env python3
"""
EKF SLAM State Animator
-----------------------
Reads build/ekf_state_log.txt and produces a matplotlib animation showing:
  - Robot pose  (triangle pointing in heading direction)
  - Robot position uncertainty ellipse  (N_SIGMA * sqrt(var_x/y))
  - Landmark positions  (cross markers, per-landmark colour)
  - Landmark uncertainty ellipses  (N_SIGMA * sqrt(var_x/y))
  - Trajectory trail

Note: the *_std_* columns in the log file are covariance diagonal elements
(variances), so sqrt() is applied before use.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Polygon
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
LOG_PATH   = Path(__file__).parent / "build" / "ekf_state_log.txt"
N_SIGMA    = 3       # ellipse radius in standard deviations
TRAIL_LEN  = 300     # max number of past poses to draw as trajectory trail
STEP       = 1       # animate every STEP-th data row (increase to speed up)
FPS        = 30      # animation frames per second
SAVE_PATH  = None    # set to e.g. "ekf_slam.mp4" to save instead of showing

LM_COLORS  = ["tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def robot_vertices(x: float, y: float, theta: float, size: float) -> np.ndarray:
    """Triangle vertices (3×2) for a robot at (x,y) heading theta."""
    tip   = np.array([np.cos(theta),             np.sin(theta)])
    left  = np.array([np.cos(theta + 2.4),       np.sin(theta + 2.4)])
    right = np.array([np.cos(theta - 2.4),       np.sin(theta - 2.4)])
    pts = np.array([x, y]) + size * np.stack([tip, 0.55 * left, 0.55 * right])
    return pts


def axis_aligned_ellipse(cx, cy, var_x, var_y, n_sigma, **kw) -> Ellipse:
    """Return an Ellipse patch with semi-axes n_sigma*sqrt(var)."""
    w = 2.0 * n_sigma * np.sqrt(max(var_x, 0.0))
    h = 2.0 * n_sigma * np.sqrt(max(var_y, 0.0))
    return Ellipse((cx, cy), width=w, height=h, **kw)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not LOG_PATH.exists():
        sys.exit(f"Log file not found: {LOG_PATH}")

    df = pd.read_csv(LOG_PATH)

    # Detect landmarks from column names  (lm0_x, lm1_x, …)
    lm_ids = sorted({
        int(c[2:c.index("_", 2)])
        for c in df.columns if c.startswith("lm") and "_x" in c and not c.endswith("_std_x")
    })
    # Safer: count columns named lm{i}_x
    import re
    lm_ids = sorted({
        int(m.group(1))
        for c in df.columns
        for m in [re.match(r"^lm(\d+)_x$", c)]
        if m
    })
    n_lm = len(lm_ids)

    # Determine which frames to animate
    frame_indices = list(range(0, len(df), STEP))
    n_frames = len(frame_indices)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("X  [m]", fontsize=11)
    ax.set_ylabel("Y  [m]", fontsize=11)
    ax.set_title("EKF SLAM — Robot & Landmark State Estimates", fontsize=13)
    ax.grid(True, alpha=0.25, lw=0.5)

    # Compute axis limits up front
    xs = list(df["robot_x"])
    ys = list(df["robot_y"])
    for i in lm_ids:
        lx = df[f"lm{i}_x"]
        ly = df[f"lm{i}_y"]
        active = (lx.abs() > 0.01) | (ly.abs() > 0.01)
        xs += list(lx[active])
        ys += list(ly[active])
    pad_x = (max(xs) - min(xs)) * 0.08 + 80
    pad_y = (max(ys) - min(ys)) * 0.08 + 80
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    map_diag  = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    robot_sz  = map_diag * 0.012   # triangle size relative to map

    # ── Static artists ────────────────────────────────────────────────────────
    # Trajectory trail
    trail_line, = ax.plot([], [], "-", color="steelblue", lw=0.9,
                          alpha=0.55, label="Robot trajectory", zorder=2)

    # Robot pose triangle
    robot_tri = Polygon(
        robot_vertices(0, 0, 0, robot_sz),
        closed=True, fc="steelblue", ec="navy", lw=1.2, zorder=6,
    )
    ax.add_patch(robot_tri)

    # Robot uncertainty ellipse
    robot_ell = Ellipse(
        (0, 0), width=1, height=1,
        fc="none", ec="steelblue", lw=1.1, ls="--", alpha=0.6, zorder=3,
    )
    ax.add_patch(robot_ell)

    # Per-landmark artists
    lm_markers  = []
    lm_ellipses = []
    lm_texts    = []
    lm_legend_handles = []

    for idx, i in enumerate(lm_ids):
        color = LM_COLORS[idx % len(LM_COLORS)]
        # cross marker
        marker, = ax.plot([], [], "x", color=color, ms=9, mew=2.2, zorder=7)
        # uncertainty ellipse
        ell = Ellipse(
            (0, 0), width=1, height=1,
            fc=color, ec=color, lw=1.2, alpha=0.18, zorder=3, visible=False,
        )
        ax.add_patch(ell)
        # label
        txt = ax.text(0, 0, f"LM {i}", fontsize=8, color=color,
                      ha="left", va="bottom", zorder=8, visible=False,
                      fontweight="bold")
        lm_markers.append(marker)
        lm_ellipses.append(ell)
        lm_texts.append(txt)
        lm_legend_handles.append(
            mpatches.Patch(color=color, label=f"Landmark {i}")
        )

    # Time counter
    time_txt = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        fontsize=10, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # Sigma label in corner
    ax.text(
        0.98, 0.02,
        f"Ellipses: {N_SIGMA}σ  (sqrt of covariance diagonal)",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color="gray",
    )

    # Legend
    legend_elems = [
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label="Robot trajectory"),
        mpatches.Patch(fc="steelblue", ec="navy", label="Robot pose"),
        mpatches.Patch(fc="none", ec="steelblue", ls="--", label="Robot uncertainty"),
    ] + lm_legend_handles
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8,
              framealpha=0.85)

    # ── Update function ───────────────────────────────────────────────────────
    all_artists = (
        [trail_line, robot_tri, robot_ell, time_txt]
        + lm_markers + lm_ellipses + lm_texts
    )

    def update(frame_no: int):
        row_idx = frame_indices[frame_no]
        row = df.iloc[row_idx]

        # --- trajectory trail ---
        start = max(0, row_idx - TRAIL_LEN)
        trail_line.set_data(
            df["robot_x"].iloc[start : row_idx + 1],
            df["robot_y"].iloc[start : row_idx + 1],
        )

        # --- robot triangle ---
        robot_tri.set_xy(
            robot_vertices(row["robot_x"], row["robot_y"],
                           row["robot_theta"], robot_sz)
        )

        # --- robot uncertainty ellipse ---
        robot_ell.set_center((row["robot_x"], row["robot_y"]))
        robot_ell.set_width( 2.0 * N_SIGMA * np.sqrt(max(row["robot_std_x"], 0)))
        robot_ell.set_height(2.0 * N_SIGMA * np.sqrt(max(row["robot_std_y"], 0)))

        # --- landmarks ---
        for idx, i in enumerate(lm_ids):
            lx   = row[f"lm{i}_x"]
            ly   = row[f"lm{i}_y"]
            vx   = row[f"lm{i}_std_x"]   # variance (despite "std" name)
            vy   = row[f"lm{i}_std_y"]

            # A landmark is "active" once it has been first observed
            # (position leaves the origin AND variance exceeds the tiny prior)
            active = (abs(lx) > 0.01 or abs(ly) > 0.01) and vx > 0.4

            if active:
                lm_markers[idx].set_data([lx], [ly])
                lm_ellipses[idx].set_center((lx, ly))
                lm_ellipses[idx].set_width( 2.0 * N_SIGMA * np.sqrt(max(vx, 0)))
                lm_ellipses[idx].set_height(2.0 * N_SIGMA * np.sqrt(max(vy, 0)))
                lm_ellipses[idx].set_visible(True)
                lm_texts[idx].set_position((lx + robot_sz * 0.6,
                                            ly + robot_sz * 0.6))
                lm_texts[idx].set_visible(True)
            else:
                lm_markers[idx].set_data([], [])
                lm_ellipses[idx].set_visible(False)
                lm_texts[idx].set_visible(False)

        time_txt.set_text(f"t = {row['time']:6.2f} s   frame {row_idx+1}/{len(df)}")
        return all_artists

    # ── Animate ───────────────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000.0 / FPS,
        blit=True,
    )

    plt.tight_layout()

    if SAVE_PATH:
        print(f"Saving animation to {SAVE_PATH} …")
        anim.save(SAVE_PATH, fps=FPS, dpi=150,
                  writer="ffmpeg" if SAVE_PATH.endswith(".mp4") else "pillow")
        print("Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
