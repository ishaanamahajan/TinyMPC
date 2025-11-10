#!/usr/bin/env python3
"""
Compare PSD-based and TV-linear obstacle avoidance trajectories.

Inputs:
  - psd_trajectory.csv     (from examples/tiny_psd_demo)
  - tv_linear_trajectory.csv (from examples/tv_linear_demo)

Outputs:
  - compare_tv_vs_psd.png  (overlay plots)
  - prints a metric table for quick inspection

Notes:
  - Cost comparison uses ONLY base-state and base-input weights so the
    comparison is fair (ignores lifted costs present only in PSD demo).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def base_cost(x1, x2, x3, x4, u1, u2, Qb=None, Rb=None):
    # Default base weights (match examples): diag(Qb)=[10,10,1,1], diag(Rb)=[2,2]
    if Qb is None:
        Qb = np.diag([10.0, 10.0, 1.0, 1.0])
    if Rb is None:
        Rb = np.diag([2.0, 2.0])
    X = np.vstack([x1, x2, x3, x4])  # shape (4, N)
    U = np.vstack([u1[:-1], u2[:-1]])  # shape (2, N-1)
    Jx = np.sum(np.einsum('ij,ji->i', X.T @ Qb, X))
    Ju = np.sum(np.einsum('ij,ji->i', U.T @ Rb, U)) if U.size else 0.0
    return Jx + Ju, Jx, Ju


def ctrl_smoothness(u1, u2):
    du1 = np.diff(u1)
    du2 = np.diff(u2)
    return float(np.sum(du1**2 + du2**2))


def summarize(df, label, obstacle=(-5.0, 0.0, 2.0)):
    k = df['k'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    x3 = df['x3'].values
    x4 = df['x4'].values
    u1 = df['u1'].values
    u2 = df['u2'].values

    ox, oy, r = obstacle
    dist = np.sqrt((x1 - ox)**2 + (x2 - oy)**2)
    min_dist = float(dist.min())
    peak_y = float(np.max(np.abs(x2)))
    goal_err = float(np.hypot(x1[-1], x2[-1]))

    J, Jx, Ju = base_cost(x1, x2, x3, x4, u1, u2)
    smooth = ctrl_smoothness(u1, u2)

    out = {
        'label': label,
        'N': len(k),
        'min_clearance': min_dist - r,
        'peak_abs_y': peak_y,
        'goal_error': goal_err,
        'base_cost': J,
        'base_cost_x': Jx,
        'base_cost_u': Ju,
        'du2_cost': smooth,
    }

    # PSD only: rank1 gap summary if present
    if 'rank1_gap' in df.columns:
        rg = df['rank1_gap'].values
        out.update({
            'rank1_gap_max': float(rg.max()),
            'rank1_gap_mean': float(rg.mean()),
            'rank1_gap_final': float(rg[-1]),
        })
    return out


def print_table(rows):
    # Minimal text table
    keys = [
        'label','N','min_clearance','peak_abs_y','goal_error',
        'base_cost','base_cost_x','base_cost_u','du2_cost',
        'rank1_gap_max','rank1_gap_mean','rank1_gap_final'
    ]
    # Determine available columns
    present = [k for k in keys if any(k in r for r in rows)]
    widths = {k: max(len(k), max(len(f"{r.get(k,'')}") for r in rows)) for k in present}
    line = ' | '.join(k.ljust(widths[k]) for k in present)
    print(line)
    print('-' * len(line))
    for r in rows:
        print(' | '.join(f"{r.get(k,'')}".ljust(widths[k]) for k in present))


def overlay_plot(df_psd, df_tv, obstacle=(-5.0, 0.0, 2.0)):
    ox, oy, r = obstacle
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 2D path
    ax = axes[0,0]
    ax.set_aspect('equal')
    theta = np.linspace(0, 2*np.pi, 256)
    ax.fill(ox + r*np.cos(theta), oy + r*np.sin(theta), color='gray', alpha=0.4, label='Obstacle')
    ax.plot(df_tv['x1'], df_tv['x2'], 'r.-', ms=3, label='TV-Linear')
    ax.plot(df_psd['x1'], df_psd['x2'], 'b.-', ms=3, label='PSD')
    ax.set_title('Trajectory (2D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # y(t)
    ax = axes[0,1]
    ax.plot(df_tv['k'], df_tv['x2'], 'r-', label='TV-Linear y')
    ax.plot(df_psd['k'], df_psd['x2'], 'b-', label='PSD y')
    ax.set_title('Position y over time')
    ax.set_xlabel('k')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # u(t)
    ax = axes[1,0]
    ax.plot(df_tv['k'][:-1], df_tv['u1'][:-1], 'r-', alpha=0.8, label='TV-Linear u1')
    ax.plot(df_tv['k'][:-1], df_tv['u2'][:-1], 'r--', alpha=0.8, label='TV-Linear u2')
    ax.plot(df_psd['k'][:-1], df_psd['u1'][:-1], 'b-', alpha=0.8, label='PSD u1')
    ax.plot(df_psd['k'][:-1], df_psd['u2'][:-1], 'b--', alpha=0.8, label='PSD u2')
    ax.set_title('Controls')
    ax.set_xlabel('k')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # Rank-1 gap (PSD only)
    ax = axes[1,1]
    if 'rank1_gap' in df_psd.columns:
        ax.plot(df_psd['k'], df_psd['rank1_gap'], 'm-', label='PSD rank-1 gap')
        ax.set_yscale('log')
    ax.set_title('PSD rank-1 gap (log)')
    ax.set_xlabel('k')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = 'compare_tv_vs_psd.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved overlay to {out}")


def main():
    psd_csv = 'psd_trajectory.csv'
    tv_csv = 'tv_linear_trajectory.csv'
    if not (os.path.exists(psd_csv) and os.path.exists(tv_csv)):
        print('Error: missing CSVs. Run tiny_psd_demo and tv_linear_demo first.')
        sys.exit(1)

    df_psd = pd.read_csv(psd_csv)
    df_tv  = pd.read_csv(tv_csv)

    rows = [
        summarize(df_tv, 'TV-Linear'),
        summarize(df_psd, 'PSD'),
    ]
    print_table(rows)
    overlay_plot(df_psd, df_tv)


if __name__ == '__main__':
    main()

