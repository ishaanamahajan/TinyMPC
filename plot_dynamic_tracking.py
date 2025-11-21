#!/usr/bin/env python3
"""
Visualize the dynamic PSD tracking run with moving obstacles.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


TRACKING_CSV = "psd_dynamic_tracking.csv"
OBSTACLE_CSV = "psd_dynamic_obstacles.csv"
PLAN_LOG_CSV = "psd_dynamic_plan_log.csv"
OUTPUT_FIG = "psd_dynamic_plots.png"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def build_obstacle_traces(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    traces: Dict[int, pd.DataFrame] = {}
    for disk_id, group in df.groupby("disk"):
        traces[int(disk_id)] = group.sort_values("k")
    return traces


def main() -> None:
    track_df = load_csv(TRACKING_CSV)
    obs_df = load_csv(OBSTACLE_CSV)
    plan_df = pd.read_csv(PLAN_LOG_CSV) if os.path.exists(PLAN_LOG_CSV) else None

    k = track_df["k"].values
    x1 = track_df["x1"].values
    x2 = track_df["x2"].values
    x3 = track_df["x3"].values
    x4 = track_df["x4"].values
    u1 = track_df["u1"].values
    u2 = track_df["u2"].values
    signed_dist = track_df["signed_dist"].values
    plan_age = track_df["plan_age"].values
    solver_iter = track_df["solver_iter"].values

    obstacle_traces = build_obstacle_traces(obs_df)
    disk_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(obstacle_traces)))

    fig, axes = plt.subplots(4, 1, figsize=(11, 16))

    # 2D layout
    ax = axes[0]
    ax.set_aspect("equal")
    ax.plot(x1, x2, "k.-", linewidth=2, label="Agent trajectory")
    ax.scatter([x1[0]], [x2[0]], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter([x1[-1]], [x2[-1]], c="red", s=120, marker="*", label="End", zorder=5)

    for (disk_id, trace), color in zip(obstacle_traces.items(), disk_colors):
        ax.plot(trace["cx"], trace["cy"], color=color, linestyle="--", label=f"Disk {disk_id} path")
        final = trace.iloc[-1]
        circle = Circle((final["cx"], final["cy"]), final["r"], color=color, alpha=0.2)
        ax.add_patch(circle)

    ax.set_title("Agent vs. Moving Obstacles")
    ax.set_xlabel("x₁ (m)")
    ax.set_ylabel("x₂ (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # States over time
    ax = axes[1]
    ax.plot(k, x1, label="x₁")
    ax.plot(k, x2, label="x₂")
    ax.plot(k, x3, label="x₃")
    ax.plot(k, x4, label="x₄")
    ax.set_title("States vs. time")
    ax.set_xlabel("Step k")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Controls overtime
    ax = axes[2]
    ax.plot(k, u1, label="u₁")
    ax.plot(k, u2, label="u₂")
    ax.set_title("Controls vs. time")
    ax.set_xlabel("Step k")
    ax.set_ylabel("Acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Signed distance and plan age
    ax = axes[3]
    ax.plot(k, signed_dist, label="Signed distance", color="tab:purple")
    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Step k")
    ax.set_title("Safety margin & plan refresh")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(k, plan_age, label="Plan age", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Plan age (steps)")

    if plan_df is not None and not plan_df.empty:
        for _, row in plan_df.iterrows():
            ax.axvline(int(row["replan_step"]), color="gray", alpha=0.2, linestyle=":")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    print(f"✅ Saved dynamic visualization to {OUTPUT_FIG}")
    print(f"Min signed distance: {signed_dist.min():.3f} m")
    print(f"Max plan age: {plan_age.max()} steps")
    print(f"Mean solver iterations: {solver_iter.mean():.1f}")


if __name__ == "__main__":
    main()


