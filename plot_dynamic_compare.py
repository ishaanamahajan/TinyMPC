#!/usr/bin/env python3
"""
Compare PSD/TV/CBF dynamic tracking runs with moving obstacles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

PSD_CSV = "psd_dynamic_tracking.csv"
TV_CSV = "tv_dynamic_tracking.csv"
CBF_CSV = "cbf_dynamic_tracking.csv"
PLAN_LOG_CSV = "psd_dynamic_plan_log.csv"
OBSTACLE_FILES = [
    "psd_dynamic_obstacles.csv",
    "tv_dynamic_obstacles.csv",
    "cbf_dynamic_obstacles.csv",
]
OUTPUT_FIG = "psd_dynamic_plots.png"
COMPARE_FIG = "dynamic_compare.png"
ANIM_OUT = "dynamic_compare.gif"


def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to load {path}: {exc}")
        return None


def build_obstacle_traces(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    traces: Dict[int, pd.DataFrame] = {}
    for disk_id, group in df.groupby("disk"):
        traces[int(disk_id)] = group.sort_values("k")
    return traces


@dataclass
class MovingDisk:
    cx0: float
    cy0: float
    vx: float
    vy: float
    radius: float
    wobble_x: float
    wobble_x_freq: float
    wobble_x_phase: float
    wobble_y: float
    wobble_y_freq: float
    wobble_y_phase: float

    def disk_at_time(self, t: float) -> Tuple[float, float, float]:
        cx = self.cx0 + self.vx * t + self.wobble_x * np.sin(self.wobble_x_freq * t + self.wobble_x_phase)
        cy = self.cy0 + self.vy * t + self.wobble_y * np.cos(self.wobble_y_freq * t + self.wobble_y_phase)
        return cx, cy, self.radius


def synthesize_obstacles(max_k: int) -> pd.DataFrame:
    disks = [
        MovingDisk(-2.5, 3.0, 0.0, -0.12, 0.9, 0.15, 0.8, 0.0, 0.05, 0.6, 0.0),
        MovingDisk(-0.5, -3.5, 0.0, 0.10, 0.8, 0.12, 0.7, 1.2, 0.04, 0.7, 0.5),
        MovingDisk(1.8, 1.5, 0.02, -0.06, 0.7, 0.1, 0.5, -0.4, 0.03, 0.5, 0.9),
    ]
    rows = []
    for k in range(max_k + 1):
        for j, disk in enumerate(disks):
            cx, cy, r = disk.disk_at_time(k)
            rows.append((k, j, cx, cy, r))
    return pd.DataFrame(rows, columns=["k", "disk", "cx", "cy", "r"])


def load_or_synthesize_obstacles(max_k: int) -> pd.DataFrame:
    for path in OBSTACLE_FILES:
        df = load_csv(path)
        if df is not None:
            return df
    print("Warning: obstacle CSV not found, synthesizing from scenario definition.")
    return synthesize_obstacles(max_k)


def main() -> None:
    runs = [
        ("PSD", PSD_CSV, "tab:red", "-"),
        ("TV", TV_CSV, "tab:green", "--"),
        ("CBF", CBF_CSV, "tab:purple", ":"),
    ]
    data = []
    max_k = 0
    for label, path, color, style in runs:
        df = load_csv(path)
        if df is None:
            print(f"Warning: {path} missing; skipping {label}.")
            continue
        data.append((label, df, color, style))
        if "k" in df.columns:
            max_k = max(max_k, int(df["k"].max()))
    if not data:
        raise SystemExit("No dynamic tracking CSVs found.")

    obs_df = load_or_synthesize_obstacles(max_k)
    obstacle_traces = build_obstacle_traces(obs_df)

    plan_log = load_csv(PLAN_LOG_CSV)

    fig, axes = plt.subplots(3, 1, figsize=(11, 14))

    # Trajectories
    ax = axes[0]
    ax.set_aspect("equal")
    for (disk_id, trace) in obstacle_traces.items():
        ax.plot(trace["cx"], trace["cy"], linestyle="--", linewidth=1.0, color="gray", alpha=0.6)
        final = trace.iloc[-1]
        ax.add_patch(plt.Circle((final["cx"], final["cy"]), final["r"], color="gray", alpha=0.15))
    for label, df, color, style in data:
        if {"x1", "x2"} <= set(df.columns):
            ax.plot(df["x1"], df["x2"], color=color, linestyle=style, linewidth=2, label=label)
        else:
            print(f"Warning: {label} CSV missing x1/x2 columns.")
    ax.scatter([-8.0], [0.0], c="black", s=80, marker="o", label="Start")
    ax.scatter([0.0], [0.0], c="gold", s=120, marker="*", label="Goal")
    ax.set_title("2D trajectories with crossing obstacles")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Signed distance
    ax = axes[1]
    for label, df, color, style in data:
        if "signed_dist" not in df.columns:
            continue
        sd_col = "seg_signed_dist" if "seg_signed_dist" in df.columns else "signed_dist"
        ax.plot(df["k"], df[sd_col], color=color, linestyle=style, linewidth=2,
                label=f"{label} (segment)")
        if sd_col != "signed_dist":
            ax.plot(df["k"], df["signed_dist"], color=color, linestyle=":",
                    linewidth=1.2, alpha=0.6, label=f"{label} (point)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Signed distance to nearest disk")
    ax.set_xlabel("Step k")
    ax.set_ylabel("signed_dist")
    if plan_log is not None:
        for _, row in plan_log.iterrows():
            ax.axvline(row["replan_step"], color="gray", linestyle=":", alpha=0.2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Progress (x1) and plan age for PSD
    ax = axes[2]
    for label, df, color, style in data:
        if "x1" not in df.columns:
            continue
        ax.plot(df["k"], df["x1"], color=color, linestyle=style, linewidth=2, label=f"{label} x₁")
    ax.set_title("Progress along x-axis")
    ax.set_xlabel("Step k")
    ax.set_ylabel("x₁")
    ax.grid(True, alpha=0.3)

    if any(label == "PSD" for label, _, _, _ in data):
        psd_df = next(df for label, df, _, _ in data if label == "PSD")
        if "plan_age" in psd_df.columns:
            ax2 = ax.twinx()
            ax2.plot(psd_df["k"], psd_df["plan_age"], color="tab:orange", linestyle="-.", linewidth=1.5,
                     label="PSD plan age")
            ax2.set_ylabel("Plan age (steps)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax.legend()
    else:
        ax.legend()

    plt.tight_layout()
    plt.savefig(COMPARE_FIG, dpi=150, bbox_inches="tight")
    print(f"✅ Dynamic comparison saved to {COMPARE_FIG}")

    # Animation setup
    anim_fig, anim_ax = plt.subplots(figsize=(8, 6))
    anim_ax.set_aspect("equal")
    anim_ax.set_title("Dynamic crossing scenario")
    anim_ax.set_xlabel("x₁")
    anim_ax.set_ylabel("x₂")
    anim_ax.grid(True, alpha=0.3)

    color_map = {"PSD": "tab:red", "TV": "tab:green", "CBF": "tab:purple"}
    traj_lines = {}
    scatters = {}
    for label, df, _, _ in data:
        color = color_map.get(label, "black")
        line, = anim_ax.plot([], [], color=color, linewidth=2, label=label)
        traj_lines[label] = line
        scatters[label] = anim_ax.scatter([], [], color=color, s=50)
    anim_ax.legend(loc="upper left")

    disk_patches = []
    for _ in obstacle_traces:
        patch = plt.Circle((0, 0), 0.0, color="gray", alpha=0.25)
        anim_ax.add_patch(patch)
        disk_patches.append(patch)

    max_frames = max_k + 1

    def init_anim():
        for line in traj_lines.values():
            line.set_data([], [])
        for scatter in scatters.values():
            scatter.set_offsets(np.empty((0, 2)))
        for patch in disk_patches:
            patch.set_radius(0.0)
        return list(traj_lines.values()) + list(scatters.values()) + disk_patches

    def update(frame):
        for label, df, _, _ in data:
            df_partial = df[df["k"] <= frame]
            traj_lines[label].set_data(df_partial["x1"], df_partial["x2"])
            if not df_partial.empty:
                last = np.array([[df_partial["x1"].iloc[-1], df_partial["x2"].iloc[-1]]])
                scatters[label].set_offsets(last)
            else:
                scatters[label].set_offsets(np.empty((0, 2)))
        for patch, (_, trace) in zip(disk_patches, obstacle_traces.items()):
            row = trace[trace["k"] == frame]
            if row.empty:
                patch.set_radius(0.0)
            else:
                patch.center = (row["cx"].iloc[0], row["cy"].iloc[0])
                patch.set_radius(row["r"].iloc[0])
        anim_ax.set_xlim(-12, 4)
        anim_ax.set_ylim(-5, 5)
        return list(traj_lines.values()) + list(scatters.values()) + disk_patches

    anim = animation.FuncAnimation(
        anim_fig,
        update,
        init_func=init_anim,
        frames=max_frames,
        interval=150,
        blit=True,
    )
    try:
        import PIL  # noqa: F401

        anim.save(ANIM_OUT, writer="pillow", dpi=100)
        print(f"🎞️ Animated visualization saved to {ANIM_OUT}")
    except ImportError:
        print("Warning: pillow not installed; skipping GIF export (pip install pillow).")
    except Exception as exc:
        print(f"Warning: failed to save GIF: {exc}")

    for label, df, _, _ in data:
        if "signed_dist" in df.columns:
            sd_col = "seg_signed_dist" if "seg_signed_dist" in df.columns else "signed_dist"
            print(f"{label}: min signed distance = {df[sd_col].min():.3f}")


if __name__ == "__main__":
    main()


