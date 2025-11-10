#!/usr/bin/env python3
"""
Plot the time‑varying linear (tangent) obstacle avoidance demo trajectory.
Reads tv_linear_trajectory.csv produced by examples/tv_linear_demo.cpp and
generates a combined figure with: 2D path, states, controls.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = 'tv_linear_trajectory.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run tv_linear_demo first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Required columns from tv_linear_demo
    required = ['k', 'x1', 'x2', 'x3', 'x4', 'u1', 'u2']
    for c in required:
        if c not in df.columns:
            print(f"Error: column '{c}' missing in {csv_path}")
            sys.exit(1)

    k = df['k'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    x3 = df['x3'].values
    x4 = df['x4'].values
    u1 = df['u1'].values
    u2 = df['u2'].values

    # Obstacle parameters (must match the demo)
    x_obs = np.array([-5.0, 0.0])
    r_obs = 2.0

    # Initial and goal (for reference)
    x_initial = np.array([-10.0, 0.1])
    x_goal = np.array([0.0, 0.0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1) 2D trajectory with obstacle
    ax = axes[0]
    ax.set_aspect('equal')
    theta = np.linspace(0, 2*np.pi, 256)
    ax.fill(x_obs[0] + r_obs*np.cos(theta), x_obs[1] + r_obs*np.sin(theta),
            color='gray', alpha=0.4, label='Obstacle')
    ax.plot(x1, x2, 'b.-', lw=2, ms=4, label='Trajectory')
    ax.scatter([x_initial[0]], [x_initial[1]], c='green', s=80, label='Start')
    ax.scatter([x_goal[0]], [x_goal[1]], c='red', s=100, marker='*', label='Goal')
    ax.set_xlabel('x₁ (position x)')
    ax.set_ylabel('x₂ (position y)')
    ax.set_title('TV-Linear Obstacle Avoidance: 2D Path')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) States over time
    ax = axes[1]
    ax.plot(k, x1, lw=2, label='x₁ (x)')
    ax.plot(k, x2, lw=2, label='x₂ (y)')
    ax.plot(k, x3, lw=2, label='x₃ (vx)')
    ax.plot(k, x4, lw=2, label='x₄ (vy)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title('States')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # 3) Controls over time
    ax = axes[2]
    ax.plot(k[:-1], u1[:-1], lw=2, label='u₁ (ax)')
    ax.plot(k[:-1], u2[:-1], lw=2, label='u₂ (ay)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Control')
    ax.set_title('Controls')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_png = 'tv_linear_plots.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plots to {out_png}")

    # Simple summary
    distances = np.sqrt((x1 - x_obs[0])**2 + (x2 - x_obs[1])**2)
    print('\nSummary:')
    print(f"  Start: ({x_initial[0]:.2f}, {x_initial[1]:.2f})  ->  End: ({x1[-1]:.4f}, {x2[-1]:.4f})")
    print(f"  Closest approach to obstacle: {distances.min():.4f} (radius {r_obs:.2f})")
    print(f"  Peak |y|: {np.max(np.abs(x2)):.4f}")


if __name__ == '__main__':
    main()

