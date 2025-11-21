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

DEFAULT_PIPELINE_CSV = "psd_tv_pipeline_stage2_tv.csv"
DEFAULT_PSD_REG_CSV = "psd_tv_combo_trajectory.csv"
DEFAULT_TV_CSV = "tv_linear_trajectory.csv"
DEFAULT_CBF_CSV = "cbf_ushape_trajectory.csv"
TRAJ_ONLY_OUT = "traj_compare.png"


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


def pos_xy(df):
    if df is None:
        return None
    cols = set(df.columns)
    if {'x_dyn', 'y_dyn'} <= cols:
        return df['x_dyn'].to_numpy(), df['y_dyn'].to_numpy(), 'dyn'
    if {'x1', 'x2'} <= cols:
        return df['x1'].to_numpy(), df['x2'].to_numpy(), 'x'
    return None


def signed_distance_series(df, obstacles):
    if df is None:
        return None
    cols = set(df.columns)
    if 'sd_dyn' in cols:
        return df['sd_dyn'].to_numpy()
    if 'sd_min' in cols:
        return df['sd_min'].to_numpy()
    if 'signed_dist' in cols:
        return df['signed_dist'].to_numpy()
    if {'x_dyn', 'y_dyn'} <= cols:
        x1 = df['x_dyn'].to_numpy()
        x2 = df['y_dyn'].to_numpy()
    elif {'x1', 'x2'} <= cols:
        x1 = df['x1'].to_numpy()
        x2 = df['x2'].to_numpy()
    else:
        return None
    all_sd = []
    for (cx, cy, rr) in obstacles:
        all_sd.append(np.sqrt((x1 - cx)**2 + (x2 - cy)**2) - rr)
    return np.min(np.vstack(all_sd), axis=0)


def seg_signed_dist_to_circle(p0, p1, c, r):
    v = p1 - p0
    vv = np.dot(v, v)
    if vv == 0.0:
        return np.linalg.norm(p0 - c) - r
    t = np.clip(np.dot(c - p0, v) / vv, 0.0, 1.0)
    closest = p0 + t * v
    return np.linalg.norm(closest - c) - r


def segmentwise_signed_distance_series(df, obstacles):
    if df is None:
        return None
    if 'x1' in df and 'x2' in df:
        x = df['x1'].to_numpy()
        y = df['x2'].to_numpy()
    elif 'x_dyn' in df and 'y_dyn' in df:
        x = df['x_dyn'].to_numpy()
        y = df['y_dyn'].to_numpy()
    else:
        return None
    if len(x) < 2:
        return None
    sds = []
    for k in range(len(x) - 1):
        p0 = np.array([x[k],   y[k]])
        p1 = np.array([x[k+1], y[k+1]])
        best = np.inf
        for (cx, cy, rr) in obstacles:
            d = seg_signed_dist_to_circle(p0, p1, np.array([cx, cy]), rr)
            if d < best:
                best = d
        sds.append(best)
    return np.array(sds)


def infer_start_from_data(data, fallback):
    def first_point(entry):
        label, df, _, _ = entry
        xy = pos_xy(df)
        if xy is None or len(xy[0]) == 0:
            return None
        return np.array([xy[0][0], xy[1][0]])

    for entry in data:
        label = entry[0]
        if 'PSD' in label.upper():
            pt = first_point(entry)
            if pt is not None:
                return pt
    for entry in data:
        pt = first_point(entry)
        if pt is not None:
            return pt
    return fallback


def save_traj_only_plot(outfile, data, obstacles, start, goal, title):
    if not data:
        return
    theta = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    for idx, (cx, cy, rr) in enumerate(obstacles):
        ax.fill(cx + rr * np.cos(theta),
                cy + rr * np.sin(theta),
                color='gray', alpha=0.35,
                label='Obstacle' if idx == 0 else None)
    has_traj = False
    for label, df, color, style in data:
        xy = pos_xy(df)
        if xy is None:
            continue
        has_traj = True
        ax.plot(xy[0], xy[1], color=color, linestyle=style, linewidth=2, label=label)
    if not has_traj:
        plt.close(fig)
        print("Warning: no trajectory data available for trajectory-only plot.")
        return
    ax.scatter([start[0]], [start[1]], c='green', s=60, marker='o', label='Start')
    ax.scatter([goal[0]], [goal[1]], c='black', s=70, marker='*', label='Goal')
    ax.set_title(title)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Trajectory-only plot saved to {outfile}")


def plot_narrow2d_overlay(args):
    runs = [
        ("PSD+TV (pipeline)", args.narrow_pipeline_csv, 'red', '-'),
        ("PSD+TV (linear)", args.narrow_psd_linear_csv, 'blue', '--'),
        ("TV-only", args.narrow_tv_csv, 'green', ':'),
    ]
    data = []
    for label, path, color, style in runs:
        df = load_csv(path)
        if df is None:
            print(f"Warning: unable to load {path} for {label}")
            continue
        data.append((label, df, color, style))
    if not data:
        print("No CSVs available for narrow2d plotting.")
        return 1

    obstacles = [(-3.0, 3.25, 3.0), (-3.0, -3.25, 3.0)]
    start = infer_start_from_data(data, np.array([-8.0, -4.0]))
    goal = np.array([0.0, 0.0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    ax = axes[0]
    ax.set_aspect('equal')
    theta = np.linspace(0, 2 * np.pi, 200)
    for idx, (cx, cy, rr) in enumerate(obstacles):
        ax.fill(cx + rr * np.cos(theta),
                cy + rr * np.sin(theta),
                color='gray', alpha=0.35,
                label='Obstacle' if idx == 0 else None)
    for label, df, color, style in data:
        xy = pos_xy(df)
        if xy is None:
            continue
        ax.plot(xy[0], xy[1], color=color, linestyle=style, linewidth=2, label=label)
    ax.scatter([start[0]], [start[1]], c='green', s=60, marker='o', label='Start')
    ax.scatter([goal[0]], [goal[1]], c='black', s=70, marker='*', label='Goal')
    ax.set_title('Narrow-2D corridor trajectories')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for label, df, color, style in data:
        sd = signed_distance_series(df, obstacles)
        if sd is None:
            continue
        k = df['k'].to_numpy() if 'k' in df.columns else np.arange(len(sd))
        ax.plot(k, sd, color=color, linestyle=style, linewidth=2, label=label)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Signed distance to corridor obstacles')
    ax.set_xlabel('k')
    ax.set_ylabel('signed_dist')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    for label, df, color, style in data:
        if {'k', 'rank1_gap'} <= set(df.columns):
            ax.plot(df['k'], df['rank1_gap'], color=color, linestyle=style, linewidth=2, label=label)
    ax.set_title('Rank-1 gap (‖XX - xxᵀ‖)')
    ax.set_xlabel('k')
    ax.set_ylabel('gap')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"✅ Narrow-2D comparison saved to {args.out}")
    save_traj_only_plot(TRAJ_ONLY_OUT, data, obstacles, start, goal, "Narrow-2D trajectories")

    for label, df, _, _ in data:
        sd = signed_distance_series(df, obstacles)
        if sd is not None:
            print(f"{label}: min signed_dist = {sd.min():.4f}")
        if {'rank1_gap'} <= set(df.columns):
            print(f"{label}: max rank1_gap = {df['rank1_gap'].max():.4f}")

    return 0


def plot_regular_overlay(args):
    use_ushape = getattr(args, "use_ushape_obstacles", False)
    include_segment = use_ushape
    if use_ushape:
        runs = [
            ("PSD", args.pipeline_csv, 'red', '-'),
            ("TV-only", args.tv_csv, 'green', '-'),
        ]
        cbf_csv = getattr(args, "cbf_csv", None)
        if cbf_csv:
            runs.append(("CBF", cbf_csv, 'purple', '-'))
    else:
        runs = [
            ("PSD+TV (pipeline)", args.pipeline_csv, 'red', '-'),
            ("PSD+TV (reg)", args.psd_reg_csv, 'blue', '--'),
            ("TV-only", args.tv_csv, 'green', ':'),
        ]

    data = []
    for label, path, color, style in runs:
        df = load_csv(path)
        if df is None:
            print(f"Warning: unable to load {path} for {label}")
            continue
        data.append((label, df, color, style))
    if not data:
        print("No CSVs available for plotting.")
        return 1

    if use_ushape:
        r_wall = 0.8
        obstacles = [
            (2.5,  0.0, r_wall),
            (2.5,  1.2, r_wall),
            (2.5, -1.2, r_wall),
            (3.8,  1.2, r_wall),
            (3.8, -1.2, r_wall),
            (5.0,  1.2, r_wall),
            (5.0, -1.2, r_wall),
        ]
        start = infer_start_from_data(data, np.array([6.0, 0.0]))
        goal = np.array([0.0, 0.0])
    else:
        obstacles = [(args.ox, args.oy, args.r)]
        start = infer_start_from_data(data, np.array([args.x0x, args.x0y]))
        goal = np.array([args.gx, args.gy])

    fig_rows = 4 if include_segment else 3
    fig, axes = plt.subplots(fig_rows, 1, figsize=(10, 12 if not include_segment else 14))

    ax = axes[0]
    ax.set_aspect('equal')
    theta = np.linspace(0, 2*np.pi, 200)
    for idx, (cx, cy, rr) in enumerate(obstacles):
        ax.fill(cx + rr*np.cos(theta), cy + rr*np.sin(theta),
                color='gray', alpha=0.35,
                label='Obstacle' if idx == 0 else None)
    for label, df, color, style in data:
        xy = pos_xy(df)
        if xy is None:
            continue
        ax.plot(xy[0], xy[1], color=color, linestyle=style, linewidth=2, label=label)
    ax.scatter([start[0]], [start[1]], c='green', s=60, marker='o', label='Start')
    ax.scatter([goal[0]], [goal[1]], c='black', s=70, marker='*', label='Goal')
    ax.set_title('2D trajectories (regular scenario)')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for label, df, color, style in data:
        sd = signed_distance_series(df, obstacles)
        if sd is None:
            continue
        k = df['k'].to_numpy() if 'k' in df.columns else np.arange(len(sd))
        ax.plot(k, sd, color=color, linestyle=style, linewidth=2, label=label)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Signed distance to obstacle')
    ax.set_xlabel('k')
    ax.set_ylabel('signed_dist')
    ax.grid(True, alpha=0.3)
    ax.legend()

    seg_ax = axes[2] if include_segment else None
    rank_ax = axes[3] if include_segment else axes[2]

    if include_segment:
        for label, df, color, style in data:
            seg_sd = segmentwise_signed_distance_series(df, obstacles)
            if seg_sd is None:
                continue
            seg_k = np.arange(len(seg_sd))
            seg_ax.plot(seg_k, seg_sd, color=color, linestyle=style, linewidth=2, label=label)
        seg_ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
        seg_ax.set_title('Segment-wise signed distance (collision check)')
        seg_ax.set_xlabel('segment k→k+1')
        seg_ax.set_ylabel('signed_dist')
        seg_ax.grid(True, alpha=0.3)
        seg_ax.legend()

    ax = rank_ax
    for label, df, color, style in data:
        if {'k', 'rank1_gap'} <= set(df.columns):
            ax.plot(df['k'], df['rank1_gap'], color=color, linestyle=style, linewidth=2, label=label)
    ax.set_title('Rank-1 gap (‖XX - xxᵀ‖)')
    ax.set_xlabel('k')
    ax.set_ylabel('gap')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to {args.out}")
    traj_title = "U-shape trajectories" if use_ushape else "Regular scenario trajectories"
    save_traj_only_plot(TRAJ_ONLY_OUT, data, obstacles, start, goal, traj_title)

    for label, df, _, _ in data:
        sd = signed_distance_series(df, obstacles)
        if sd is not None:
            print(f"{label}: min signed_dist = {sd.min():.4f}")
        if 'rank1_gap' in df.columns:
            print(f"{label}: max rank1_gap = {df['rank1_gap'].max():.4f}")
        seg_sd = segmentwise_signed_distance_series(df, obstacles) if include_segment else None
        if seg_sd is not None:
            print(f"{label}: min segment signed_dist = {seg_sd.min():.4f}")

    return 0


def main():
    ap = argparse.ArgumentParser(description="Compare PSD and TV-LIN trajectories")
    ap.add_argument("--pipeline_csv", default=DEFAULT_PIPELINE_CSV,
                    help="PSD→TV pipeline CSV (regular scenario)")
    ap.add_argument("--psd_reg_csv", default=DEFAULT_PSD_REG_CSV,
                    help="PSD-regularized TV CSV (regular scenario)")
    ap.add_argument("--tv_csv", default=DEFAULT_TV_CSV,
                    help="TV-only CSV (regular scenario)")
    ap.add_argument("--out", default="psd_vs_tv_plots.png", help="Output figure path")
    ap.add_argument("--narrow2d", action="store_true",
                    help="Plot narrow-2D corridor runs (pipeline vs PSD+TV linear vs TV-only)")
    ap.add_argument("--narrow_2d", action="store_true",
                    help="Alias for --narrow2d")
    ap.add_argument("--narrow_pipeline_csv", default="psd_tv_pipeline_narrow2d_stage2_tv.csv",
                    help="Narrow2D pipeline CSV (used with --narrow2d)")
    ap.add_argument("--narrow_psd_linear_csv", default="psd_tv_linear_narrow2d_trajectory.csv",
                    help="Narrow2D PSD+TV linear CSV (used with --narrow2d)")
    ap.add_argument("--narrow_tv_csv", default="tv_narrow2d_trajectory.csv",
                    help="Narrow2D TV-only CSV (used with --narrow2d)")
    ap.add_argument("--ushape", action="store_true",
                    help="Use U-shape PSD/TV CSVs")
    ap.add_argument("--u", action="store_true",
                    help="Alias for --ushape")
    ap.add_argument("--cbf_csv", default=DEFAULT_CBF_CSV,
                    help="CBF CSV (used with --ushape comparisons)")
    ap.add_argument("--ox", type=float, default=-5.0, help="Obstacle center x")
    ap.add_argument("--oy", type=float, default=0.0, help="Obstacle center y")
    ap.add_argument("--r", type=float, default=2.0, help="Obstacle radius")
    ap.add_argument("--x0x", type=float, default=-10.0, help="Initial x")
    ap.add_argument("--x0y", type=float, default=0.1, help="Initial y")
    ap.add_argument("--gx", type=float, default=0.0, help="Goal x")
    ap.add_argument("--gy", type=float, default=0.0, help="Goal y")
    args = ap.parse_args()

    use_ushape = args.ushape or args.u

    if args.narrow2d or args.narrow_2d:
        return plot_narrow2d_overlay(args)

    if use_ushape:
        if args.pipeline_csv == DEFAULT_PIPELINE_CSV:
            args.pipeline_csv = "psd_ushape_trajectory.csv"
        if args.psd_reg_csv == DEFAULT_PSD_REG_CSV:
            args.psd_reg_csv = "psd_ushape_trajectory.csv"
        if args.tv_csv == DEFAULT_TV_CSV:
            args.tv_csv = "tv_ushape_trajectory.csv"
        args.use_ushape_obstacles = True
    else:
        args.use_ushape_obstacles = False

    return plot_regular_overlay(args)

if __name__ == "__main__":
    raise SystemExit(main())
