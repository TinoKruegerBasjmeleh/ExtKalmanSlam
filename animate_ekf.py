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

Both the undisturbed (ideal) and noisy states are shown side by side.

Note: the *_std_* columns in the log file are covariance diagonal elements
(variances), so sqrt() is applied before use.
"""

import sys
import re
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

# Colours for the two robot states
IDEAL_COLOR = "steelblue"
NOISY_COLOR = "tomato"

# ── Helpers ────────────────────────────────────────────────────────────────────

def robot_vertices(x: float, y: float, theta: float, size: float) -> np.ndarray:
    """Triangle vertices (3×2) for a robot at (x,y) heading theta."""
    tip   = np.array([np.cos(theta),       np.sin(theta)])
    left  = np.array([np.cos(theta + 2.4), np.sin(theta + 2.4)])
    right = np.array([np.cos(theta - 2.4), np.sin(theta - 2.4)])
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

    # Detect landmark IDs from ideal columns (lm0_x, lm1_x, …)
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
    ax.set_title("EKF SLAM — Ideal (blue) vs Noisy (red) State Estimates",
                 fontsize=13)
    ax.grid(True, alpha=0.25, lw=0.5)

    # Compute axis limits from both ideal and noisy robot + landmark positions
    xs, ys = [], []
    for prefix in ("", "noisy_"):
        xs += list(df[f"{prefix}robot_x"])
        ys += list(df[f"{prefix}robot_y"])
        for i in lm_ids:
            lx = df[f"{prefix}lm{i}_x"]
            ly = df[f"{prefix}lm{i}_y"]
            active = (lx.abs() > 0.01) | (ly.abs() > 0.01)
            xs += list(lx[active])
            ys += list(ly[active])

    pad_x = (max(xs) - min(xs)) * 0.08 + 80
    pad_y = (max(ys) - min(ys)) * 0.08 + 80
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    map_diag = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    robot_sz = map_diag * 0.012

    # ── Static artists ────────────────────────────────────────────────────────

    # --- Ideal robot ---
    ideal_trail, = ax.plot([], [], "-", color=IDEAL_COLOR, lw=0.9,
                           alpha=0.55, label="Ideal trajectory", zorder=2)
    ideal_tri = Polygon(
        robot_vertices(0, 0, 0, robot_sz),
        closed=True, fc=IDEAL_COLOR, ec="navy", lw=1.2, zorder=6,
    )
    ax.add_patch(ideal_tri)
    ideal_ell = Ellipse(
        (0, 0), width=1, height=1,
        fc="none", ec=IDEAL_COLOR, lw=1.1, ls="--", alpha=0.6, zorder=3,
    )
    ax.add_patch(ideal_ell)

    # --- Noisy robot ---
    noisy_trail, = ax.plot([], [], "-", color=NOISY_COLOR, lw=0.9,
                           alpha=0.55, label="Noisy trajectory", zorder=2)
    noisy_tri = Polygon(
        robot_vertices(0, 0, 0, robot_sz),
        closed=True, fc=NOISY_COLOR, ec="darkred", lw=1.2, zorder=6,
    )
    ax.add_patch(noisy_tri)
    noisy_ell = Ellipse(
        (0, 0), width=1, height=1,
        fc="none", ec=NOISY_COLOR, lw=1.1, ls="--", alpha=0.6, zorder=3,
    )
    ax.add_patch(noisy_ell)

    # --- Per-landmark artists (ideal: x marker / noisy: + marker) ---
    ideal_lm_markers  = []
    ideal_lm_ellipses = []
    ideal_lm_texts    = []
    noisy_lm_markers  = []
    noisy_lm_ellipses = []
    lm_legend_handles = []

    for idx, i in enumerate(lm_ids):
        color = LM_COLORS[idx % len(LM_COLORS)]

        # Ideal landmark
        m_ideal, = ax.plot([], [], "x", color=color, ms=9, mew=2.2, zorder=7)
        e_ideal = Ellipse(
            (0, 0), width=1, height=1,
            fc=color, ec=color, lw=1.2, alpha=0.18, zorder=3, visible=False,
        )
        ax.add_patch(e_ideal)
        t_ideal = ax.text(0, 0, f"LM {i}", fontsize=8, color=color,
                          ha="left", va="bottom", zorder=8, visible=False,
                          fontweight="bold")

        # Noisy landmark ('+' marker, slightly transparent)
        m_noisy, = ax.plot([], [], "+", color=color, ms=9, mew=2.2,
                           alpha=0.55, zorder=7)
        e_noisy = Ellipse(
            (0, 0), width=1, height=1,
            fc=color, ec=color, lw=1.0, ls=":", alpha=0.12, zorder=3,
            visible=False,
        )
        ax.add_patch(e_noisy)

        ideal_lm_markers.append(m_ideal)
        ideal_lm_ellipses.append(e_ideal)
        ideal_lm_texts.append(t_ideal)
        noisy_lm_markers.append(m_noisy)
        noisy_lm_ellipses.append(e_noisy)

        lm_legend_handles.append(
            mpatches.Patch(color=color, label=f"Landmark {i}")
        )

    # Time counter
    time_txt = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        fontsize=10, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    ax.text(
        0.98, 0.02,
        f"Ellipses: {N_SIGMA}σ  (sqrt of covariance diagonal)\n"
        f"× = ideal landmark   + = noisy landmark",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color="gray",
    )

    # Legend
    legend_elems = [
        plt.Line2D([0], [0], color=IDEAL_COLOR, lw=1.5,
                   label="Ideal trajectory"),
        mpatches.Patch(fc=IDEAL_COLOR, ec="navy",   label="Ideal robot pose"),
        mpatches.Patch(fc="none",      ec=IDEAL_COLOR, ls="--",
                       label="Ideal uncertainty"),
        plt.Line2D([0], [0], color=NOISY_COLOR, lw=1.5,
                   label="Noisy trajectory"),
        mpatches.Patch(fc=NOISY_COLOR, ec="darkred", label="Noisy robot pose"),
        mpatches.Patch(fc="none",      ec=NOISY_COLOR, ls="--",
                       label="Noisy uncertainty"),
    ] + lm_legend_handles
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8,
              framealpha=0.85)

    # ── Update function ───────────────────────────────────────────────────────
    all_artists = (
        [ideal_trail, ideal_tri, ideal_ell,
         noisy_trail, noisy_tri, noisy_ell,
         time_txt]
        + ideal_lm_markers + ideal_lm_ellipses + ideal_lm_texts
        + noisy_lm_markers + noisy_lm_ellipses
    )

    def update_lm(markers, ellipses, texts, prefix, row, show_text):
        for idx, i in enumerate(lm_ids):
            lx = row[f"{prefix}lm{i}_x"]
            ly = row[f"{prefix}lm{i}_y"]
            vx = row[f"{prefix}lm{i}_std_x"]
            vy = row[f"{prefix}lm{i}_std_y"]
            active = (abs(lx) > 0.01 or abs(ly) > 0.01) and vx > 0.4
            if active:
                markers[idx].set_data([lx], [ly])
                ellipses[idx].set_center((lx, ly))
                ellipses[idx].set_width( 2.0 * N_SIGMA * np.sqrt(max(vx, 0)))
                ellipses[idx].set_height(2.0 * N_SIGMA * np.sqrt(max(vy, 0)))
                ellipses[idx].set_visible(True)
                if texts is not None:
                    texts[idx].set_position(
                        (lx + robot_sz * 0.6, ly + robot_sz * 0.6))
                    texts[idx].set_visible(True)
            else:
                markers[idx].set_data([], [])
                ellipses[idx].set_visible(False)
                if texts is not None:
                    texts[idx].set_visible(False)

    def update(frame_no: int):
        row_idx = frame_indices[frame_no]
        row = df.iloc[row_idx]
        start = max(0, row_idx - TRAIL_LEN)

        # --- Ideal robot ---
        ideal_trail.set_data(
            df["robot_x"].iloc[start : row_idx + 1],
            df["robot_y"].iloc[start : row_idx + 1],
        )
        ideal_tri.set_xy(
            robot_vertices(row["robot_x"], row["robot_y"],
                           row["robot_theta"], robot_sz)
        )
        ideal_ell.set_center((row["robot_x"], row["robot_y"]))
        ideal_ell.set_width( 2.0 * N_SIGMA * np.sqrt(max(row["robot_std_x"], 0)))
        ideal_ell.set_height(2.0 * N_SIGMA * np.sqrt(max(row["robot_std_y"], 0)))

        # --- Noisy robot ---
        noisy_trail.set_data(
            df["noisy_robot_x"].iloc[start : row_idx + 1],
            df["noisy_robot_y"].iloc[start : row_idx + 1],
        )
        noisy_tri.set_xy(
            robot_vertices(row["noisy_robot_x"], row["noisy_robot_y"],
                           row["noisy_robot_theta"], robot_sz)
        )
        noisy_ell.set_center((row["noisy_robot_x"], row["noisy_robot_y"]))
        noisy_ell.set_width(
            2.0 * N_SIGMA * np.sqrt(max(row["noisy_robot_std_x"], 0)))
        noisy_ell.set_height(
            2.0 * N_SIGMA * np.sqrt(max(row["noisy_robot_std_y"], 0)))

        # --- Landmarks ---
        update_lm(ideal_lm_markers, ideal_lm_ellipses, ideal_lm_texts,
                  "", row, show_text=True)
        update_lm(noisy_lm_markers, noisy_lm_ellipses, None,
                  "noisy_", row, show_text=False)

        time_txt.set_text(
            f"t = {row['time']:6.2f} s   frame {row_idx+1}/{len(df)}")
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
