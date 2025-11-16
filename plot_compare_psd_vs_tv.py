#!/usr/bin/env python3
"""
Compare PSD-constrained vs time-varying linear (TV-LIN) avoidance trajectories.

Reads:
  - psd_trajectory.csv (from examples/tiny_psd_demo.cpp)
  - tv_linear_trajectory.csv (from examples/tiny_tv_linear_demo.cpp)

Produces an overlay plot of 2D paths, signed distance over time, and
optionally the PSD rank-1 gap.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"Warning: file not found: {path}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Compare PSD and TV-LIN trajectories")
    ap.add_argument("--psd", default="psd_trajectory.csv", help="PSD CSV path")
    ap.add_argument("--tv", default="tv_linear_trajectory.csv", help="TV-LIN CSV path")
    ap.add_argument("--out", default="psd_vs_tv_plots.png", help="Output figure path")
    ap.add_argument("--ox", type=float, default=-5.0, help="Obstacle center x")
    ap.add_argument("--oy", type=float, default=0.0, help="Obstacle center y")
    ap.add_argument("--r", type=float, default=2.0, help="Obstacle radius")
    ap.add_argument("--x0x", type=float, default=-10.0, help="Initial x")
    ap.add_argument("--x0y", type=float, default=0.1, help="Initial y")
    ap.add_argument("--gx", type=float, default=0.0, help="Goal x")
    ap.add_argument("--gy", type=float, default=0.0, help="Goal y")
    args = ap.parse_args()

    df_psd = load_csv(args.psd)
    if df_psd is None and args.psd == "psd_trajectory.csv":
        df_psd = load_csv(Path("build")/"psd_trajectory.csv")
    df_tv = load_csv(args.tv)
    if df_tv is None and args.tv == "tv_linear_trajectory.csv":
        df_tv = load_csv(Path("build")/"tv_linear_trajectory.csv")

    if df_psd is None and df_tv is None:
        print("No input CSVs available. Run the demos first.")
        return 1

    # Prepare figure
    fig_rows = 3  # Overlay, Signed Distance, Iterations
    fig, axes = plt.subplots(fig_rows, 1, figsize=(10, 12))

    # Common obstacle and endpoints
    x_obs = np.array([args.ox, args.oy])
    r_obs = args.r
    x0 = np.array([args.x0x, args.x0y])
    xg = np.array([args.gx, args.gy])

    # Top: 2D position overlay
    ax = axes[0]
    ax.set_aspect('equal')
    th = np.linspace(0, 2*np.pi, 200)
    ax.fill(x_obs[0] + r_obs*np.cos(th), x_obs[1] + r_obs*np.sin(th),
            color='gray', alpha=0.4, label='Obstacle')

    if df_tv is not None:
        ax.plot(df_tv['x1'], df_tv['x2'], 'r.-', label='TV-LIN', linewidth=2, markersize=3)
    if df_psd is not None:
        ax.plot(df_psd['x1'], df_psd['x2'], 'b.-', label='PSD', linewidth=2, markersize=3)

    ax.scatter([x0[0]], [x0[1]], c='green', s=60, marker='o', label='Start')
    ax.scatter([xg[0]], [xg[1]], c='black', s=70, marker='*', label='Goal')
    ax.set_title('Trajectory overlay')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Helper to get signed distance (compute on the fly if missing)
    def get_sd(df):
        if df is None:
            return None
        if 'signed_dist' in df.columns:
            return df['signed_dist'].to_numpy()
        if 'x1' in df.columns and 'x2' in df.columns:
            x1 = df['x1'].to_numpy(); x2 = df['x2'].to_numpy()
            return np.sqrt((x1-args.ox)**2 + (x2-args.oy)**2) - args.r
        return None

    # Middle: signed distance over time
    ax = axes[1]
    sd_tv = get_sd(df_tv)
    sd_psd = get_sd(df_psd)
    if df_tv is not None and sd_tv is not None:
        ax.plot(df_tv['k'], sd_tv, 'r-', label='TV-LIN')
    if df_psd is not None and sd_psd is not None:
        ax.plot(df_psd['k'], sd_psd, 'b--', label='PSD')
    ax.axhline(0.0, color='k', linestyle='--', linewidth=1)
    ax.set_title('Signed distance to obstacle (>= 0 is feasible)')
    ax.set_xlabel('k')
    ax.set_ylabel('signed_dist')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Bottom: iterations (bar chart)
    ax = axes[2]
    methods = []
    iters = []
    colors = []
    if df_tv is not None and 'iter' in df_tv.columns:
        methods.append('TV-LIN'); iters.append(int(np.unique(df_tv['iter'])[0])); colors.append('red')
    if df_psd is not None and 'iter' in df_psd.columns:
        methods.append('PSD'); iters.append(int(np.unique(df_psd['iter'])[0])); colors.append('blue')
    if len(methods) > 0:
        ax.bar(methods, iters, color=colors)
        for i, v in enumerate(iters):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    ax.set_title('ADMM iterations to converge')
    ax.set_ylabel('iterations')
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to {args.out}")

    # Quick text summary
    if sd_tv is not None:
        print(f"TV-LIN: min signed_dist = {np.min(sd_tv):.6f}")
    else:
        print("TV-LIN: signed_dist unavailable")
    if sd_psd is not None:
        print(f"PSD   : min signed_dist = {np.min(sd_psd):.6f}")
    if df_tv is not None and 'iter' in df_tv.columns:
        print(f"TV-LIN: iterations = {int(np.unique(df_tv['iter'])[0])}")
    if df_psd is not None and 'iter' in df_psd.columns:
        print(f"PSD   : iterations = {int(np.unique(df_psd['iter'])[0])}")
    else:
        print("PSD   : signed_dist unavailable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
