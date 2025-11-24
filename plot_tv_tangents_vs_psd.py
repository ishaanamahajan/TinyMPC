#!/usr/bin/env python3
"""
Visualize TV tangents vs true disks and PSD planned vs executed paths.

Outputs:
- tv_dynamic_tangents_k*.png and tv_ushape_tangents_k*.png snapshots.
- psd_plan_vs_exec_k*.png overlays of planned PSD horizons vs execution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def segment_signed_distance(p0: np.ndarray, p1: np.ndarray, disks: np.ndarray) -> float:
    best = np.inf
    if disks.size == 0:
        return best
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    len2 = dx * dx + dy * dy
    for cx, cy, r in disks:
        if len2 > 0.0:
            t = ((cx - x0) * dx + (cy - y0) * dy) / len2
            t = np.clip(t, 0.0, 1.0)
        else:
            t = 0.0
        px = x0 + t * dx
        py = y0 + t * dy
        sd = math.hypot(px - cx, py - cy) - r
        if sd < best:
            best = sd
    return best


def find_tv_dynamic_k(tv: pd.DataFrame, obs: pd.DataFrame) -> int:
    max_k = int(tv["k"].max())
    for k in range(max_k):
        row0 = tv[tv["k"] == k]
        row1 = tv[tv["k"] == k + 1]
        if row0.empty or row1.empty:
            continue
        p0 = row0[["x1", "x2"]].to_numpy().ravel()
        p1 = row1[["x1", "x2"]].to_numpy().ravel()
        disks = obs[obs["k"] == k + 1][["cx", "cy", "r"]].to_numpy()
        sd = segment_signed_distance(p0, p1, disks)
        if sd < 0.0:
            return k
    # fallback to worst step
    worst_k = 0
    worst_sd = np.inf
    for k in range(max_k):
        row0 = tv[tv["k"] == k]
        row1 = tv[tv["k"] == k + 1]
        if row0.empty or row1.empty:
            continue
        p0 = row0[["x1", "x2"]].to_numpy().ravel()
        p1 = row1[["x1", "x2"]].to_numpy().ravel()
        disks = obs[obs["k"] == k + 1][["cx", "cy", "r"]].to_numpy()
        sd = segment_signed_distance(p0, p1, disks)
        if sd < worst_sd:
            worst_sd = sd
            worst_k = k
    return worst_k


def find_tv_ushape_k(tv: pd.DataFrame, disks: np.ndarray) -> int:
    best_k = 0
    best_sd = np.inf
    for _, row in tv.iterrows():
        sd = segment_signed_distance(
            np.array([row["x1"], row["x2"]]),
            np.array([row["x1"], row["x2"]]),
            disks,
        )
        if sd < best_sd:
            best_sd = sd
            best_k = int(row["k"])
    return best_k


def plot_tangents_snapshot(
    out_path: Path,
    disks: np.ndarray,
    tangents: pd.DataFrame,
    x_segment: Tuple[np.ndarray, np.ndarray],
    title: str,
) -> None:
    ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")

    if disks.size > 0:
        for cx, cy, r in disks:
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.fill(
                cx + r * np.cos(theta),
                cy + r * np.sin(theta),
                color="lightgray",
                alpha=0.5,
            )

    if tangents.empty:
        ax.plot(
            [x_segment[0][0], x_segment[1][0]],
            [x_segment[0][1], x_segment[1][1]],
            "g-o",
            label="TV segment",
        )
        ax.set_title(title + " (no tangents logged)")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    bounds = np.array(disks[:, :2]) if disks.size > 0 else np.array([[0.0, 0.0]])
    pts = np.vstack([bounds, x_segment[0][None, :], x_segment[1][None, :]])
    xmin, ymin = pts.min(axis=0) - 1.0
    xmax, ymax = pts.max(axis=0) + 1.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    shade_depth = max(xmax - xmin, ymax - ymin)
    span = shade_depth

    # Use the endpoint of the segment as a reference; this point must be feasible
    # for the stage we logged. Flip the half-space so the reference lies inside.
    ref = x_segment[1]
    for _, row in tangents.iterrows():
        a0 = row["a0"]
        a1 = row["a1"]
        b = row["b"]
        norm = math.hypot(a0, a1)
        if norm == 0:
            continue
        n_hat = np.array([a0, a1]) / norm
        s = a0 * ref[0] + a1 * ref[1] - b
        if s > 0:
            n_hat = -n_hat
        d_vec = np.array([-n_hat[1], n_hat[0]])
        p = n_hat * (b / norm)
        line_pts = np.vstack(
            [p + d_vec * span, p - d_vec * span]
        )
        ax.plot(line_pts[:, 0], line_pts[:, 1], color="tab:blue", linewidth=1.2)
        shade = np.vstack(
            [
                line_pts[0],
                line_pts[1],
                line_pts[1] - n_hat * shade_depth,
                line_pts[0] - n_hat * shade_depth,
            ]
        )
        ax.fill(
            shade[:, 0],
            shade[:, 1],
            color="tab:blue",
            alpha=0.08,
            edgecolor="none",
        )

    ax.plot(
        [x_segment[0][0], x_segment[1][0]],
        [x_segment[0][1], x_segment[1][1]],
        "g-o",
        label="TV segment",
    )
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_tv_dynamic() -> None:
    try:
        tv = pd.read_csv("tv_dynamic_tracking.csv")
        tans = pd.read_csv("tv_dynamic_tangents.csv")
        obs = pd.read_csv("tv_dynamic_obstacles.csv")
    except FileNotFoundError:
        print("[TV] Dynamic CSVs missing; skipping dynamic tangent plot.")
        return

    if tv.empty or tans.empty or obs.empty:
        print("[TV] Dynamic CSVs empty; skipping dynamic tangent plot.")
        return

    def save_snapshot(step_idx: int, label: str = "") -> None:
        row0 = tv[tv["k"] == step_idx]
        row1 = tv[tv["k"] == step_idx + 1]
        if row0.empty or row1.empty:
            print(f"[TV] Skipping k={step_idx}: insufficient trajectory data.")
            return
        # Use tangents from the solve at this outer step (k = step_idx),
        # but measure tunnelling against the disks at k+1, which is how
        # the segment-signed-distance is defined in the dynamic demo.
        disks = obs.loc[obs["k"] == step_idx + 1, ["cx", "cy", "r"]].to_numpy()
        tan_k = tans[(tans["k"] == step_idx) & (tans["stage"] == 1)]
        if tan_k.empty:
            tan_k = tans[(tans["k"] == step_idx) & (tans["stage"] == 0)]
        xk = row0[["x1", "x2"]].to_numpy().ravel()
        xk1 = row1[["x1", "x2"]].to_numpy().ravel()
        suffix = f"_{label}" if label else ""
        out_path = ROOT / f"tv_dynamic_tangents_k{step_idx}{suffix}.png"
        plot_tangents_snapshot(
            out_path,
            disks,
            tan_k,
            (xk, xk1),
            f"TV dynamic tangents @ step {step_idx}",
        )
        print(f"[TV] Dynamic tangent snapshot saved to {out_path.name}")

    targets = [0, 1, 2]
    for step in targets:
        save_snapshot(step)

    k_star = find_tv_dynamic_k(tv, obs)
    if k_star not in targets:
        save_snapshot(k_star, "worst")


def run_tv_ushape() -> None:
    try:
        tv = pd.read_csv("tv_ushape_trajectory.csv")
        tans = pd.read_csv("tv_ushape_tangents.csv")
    except FileNotFoundError:
        print("[TV] U-shape CSVs missing; skipping U-shape tangent plot.")
        return

    if tv.empty or tans.empty:
        print("[TV] U-shape CSVs empty; skipping U-shape tangent plot.")
        return

    r_wall = 0.8
    disks = np.array(
        [
            [2.5, 0.0, r_wall],
            [2.5, 1.2, r_wall],
            [2.5, -1.2, r_wall],
            [3.8, 1.2, r_wall],
            [3.8, -1.2, r_wall],
            [5.0, 1.2, r_wall],
            [5.0, -1.2, r_wall],
        ]
    )
    k_star = find_tv_ushape_k(tv, disks)
    row = tv[tv["k"] == k_star]
    if row.empty:
        print("[TV] Failed to locate step in U-shape CSV.")
        return
    x = row.iloc[0]
    x_state = np.array([x["x1"], x["x2"]])
    tan_k = tans[(tans["k"] == k_star) & (tans["stage"] == 0)]
    out_path = ROOT / f"tv_ushape_tangents_k{k_star}.png"
    plot_tangents_snapshot(
        out_path,
        disks,
        tan_k,
        (x_state, x_state),
        f"TV U-shape tangents @ step {k_star}",
    )
    print(f"[TV] U-shape tangent snapshot saved to {out_path.name}")


def plot_psd_plan_vs_exec() -> None:
    try:
        plans = pd.read_csv("psd_dynamic_plans.csv")
        track = pd.read_csv("psd_dynamic_tracking.csv")
        obs = pd.read_csv("psd_dynamic_obstacles.csv")
    except FileNotFoundError:
        print("[PSD] Required CSVs missing; skipping PSD plan plots.")
        return

    if plans.empty or track.empty:
        print("[PSD] Plans/tracking CSVs empty; skipping PSD plan plots.")
        return

    plan_groups = {k: v for k, v in plans.groupby("replan_step")}
    requested_steps = [0, 5, 10, 15]
    available_steps = [k for k in requested_steps if k in plan_groups]
    if not available_steps:
        print("[PSD] Requested replan steps missing; skipping PSD plan plots.")
        return

    horizon = int(plans["i"].max()) if "i" in plans.columns else 45
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        squeeze=False,
    )

    for ax, rstep in zip(axes.flat, available_steps):
        plan_group = plan_groups[rstep]
        ax.set_aspect("equal")
        disks = obs.loc[obs["k"] == rstep, ["cx", "cy", "r"]].to_numpy()
        if disks.size > 0:
            for cx, cy, r in disks:
                theta = np.linspace(0, 2 * np.pi, 200)
                ax.fill(
                    cx + r * np.cos(theta),
                    cy + r * np.sin(theta),
                    color="lightgray",
                    alpha=0.5,
                )

        seg = track[
            (track["k"] >= rstep) & (track["k"] <= rstep + horizon)
        ]
        max_exec_k = seg["k"].max() if not seg.empty else rstep
        exec_len = max_exec_k - rstep
        plan_subset = plan_group[plan_group["i"] <= exec_len]
        if plan_subset.empty:
            plan_subset = plan_group

        ax.plot(
            plan_subset["x1"],
            plan_subset["x2"],
            "r-",
            label=f"Plan k={rstep}",
        )
        ax.plot(
            seg["x1"],
            seg["x2"],
            "k.--",
            label="Executed",
        )
        ax.set_title(f"replan_step={rstep}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    for ax in axes.flat[len(available_steps) :]:
        ax.remove()

    fig.suptitle("PSD plan vs executed trajectories", fontsize=16)
    out_path = ROOT / "psd_plan_vs_exec_all.png"
    ensure_dir(out_path)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PSD] Plan grid saved to {out_path.name}")


def main() -> None:
    run_tv_dynamic()
    run_tv_ushape()
    plot_psd_plan_vs_exec()


if __name__ == "__main__":
    main()
