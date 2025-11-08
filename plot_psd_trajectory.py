#!/usr/bin/env python3
"""
Plot PSD-based trajectory optimization results with collision avoidance.
Mirrors the visualization from julia_sdp.jl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trajectory data
df = pd.read_csv('psd_trajectory.csv')

# Extract data
k = df['k'].values
x1 = df['x1'].values  # position x
x2 = df['x2'].values  # position y
x3 = df['x3'].values  # velocity x
x4 = df['x4'].values  # velocity y
u1 = df['u1'].values  # acceleration x
u2 = df['u2'].values  # acceleration y
rank1_gap = df['rank1_gap'].values

# Obstacle parameters (from demo)
x_obs = np.array([-5.0, 0.0])
r_obs = 2.0

# Initial and final positions
x_initial = np.array([-10.0, 0.1])
x_final = np.array([0.0, 0.0])

# Create figure with 4 subplots (matching Julia layout)
fig, axes = plt.subplots(4, 1, figsize=(10, 14))

# ==================== Plot 1: 2D Trajectory with Obstacle ====================
ax = axes[0]
ax.set_aspect('equal')

# Draw obstacle circle
theta = np.linspace(0, 2*np.pi, 200)
x_circle = x_obs[0] + r_obs * np.cos(theta)
y_circle = x_obs[1] + r_obs * np.sin(theta)
ax.fill(x_circle, y_circle, color='gray', alpha=0.5, label='Obstacle')

# Plot trajectory
ax.plot(x1, x2, 'b.-', linewidth=2, markersize=4, label='Trajectory')
ax.scatter([x_initial[0]], [x_initial[1]], s=100, c='green', marker='o', 
           zorder=5, label='Initial position')
ax.scatter([x_final[0]], [x_final[1]], s=100, c='red', marker='*', 
           zorder=5, label='Goal position')

ax.set_xlabel('x₁ (position x)')
ax.set_ylabel('x₂ (position y)')
ax.set_title('Double Integrator Trajectory with Collision Avoidance')
ax.legend()
ax.grid(True, alpha=0.3)

# ==================== Plot 2: States Over Time ====================
ax = axes[1]
ax.plot(k, x1, linewidth=2, label='x₁ (position x)')
ax.plot(k, x2, linewidth=2, label='x₂ (position y)')
ax.plot(k, x3, linewidth=2, label='x₃ (velocity x)')
ax.plot(k, x4, linewidth=2, label='x₄ (velocity y)')

ax.set_xlabel('Time Step')
ax.set_ylabel('State Value')
ax.set_title('States (x)')
ax.legend()
ax.grid(True, alpha=0.3)

# ==================== Plot 3: Controls Over Time ====================
ax = axes[2]
ax.plot(k[:-1], u1[:-1], linewidth=2, label='u₁ (acceleration x)')
ax.plot(k[:-1], u2[:-1], linewidth=2, label='u₂ (acceleration y)')

ax.set_xlabel('Time Step')
ax.set_ylabel('Control Value')
ax.set_title('Controls (u)')
ax.legend()
ax.grid(True, alpha=0.3)

# ==================== Plot 4: Rank-1 Gap (Residual Check) ====================
ax = axes[3]
ax.plot(k, rank1_gap, linewidth=2, label='‖XX - xx^T‖_F', color='purple')

ax.set_xlabel('Time Step')
ax.set_ylabel('‖X - x⊗x\'‖')
ax.set_title('Rank-1 Gap Check (SDP Relaxation Tightness)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# ==================== Save Figure ====================
plt.tight_layout()
plt.savefig('psd_combined_plots.png', dpi=150, bbox_inches='tight')
print("✅ Plots saved to psd_combined_plots.png")

# Print summary statistics
print("\n" + "="*60)
print("TRAJECTORY SUMMARY")
print("="*60)
print(f"Initial position: ({x_initial[0]:.1f}, {x_initial[1]:.1f})")
print(f"Final position:   ({x1[-1]:.4f}, {x2[-1]:.4f})")
print(f"Goal position:    ({x_final[0]:.1f}, {x_final[1]:.1f})")
print(f"Distance to goal: {np.sqrt((x1[-1]-x_final[0])**2 + (x2[-1]-x_final[1])**2):.4f}")
print(f"\nObstacle center:  ({x_obs[0]:.1f}, {x_obs[1]:.1f})")
print(f"Obstacle radius:  {r_obs:.1f}")

# Check minimum distance to obstacle
distances = np.sqrt((x1 - x_obs[0])**2 + (x2 - x_obs[1])**2)
min_dist = np.min(distances)
min_idx = np.argmin(distances)
print(f"\nClosest approach: {min_dist:.4f} at k={min_idx}")
print(f"Safety margin:    {min_dist - r_obs:.4f} (should be ≥ 0)")

# Rank-1 gap statistics
print(f"\nRank-1 gap (XX ≈ xx^T):")
print(f"  Maximum:  {np.max(rank1_gap):.4f}")
print(f"  Final:    {rank1_gap[-1]:.4f}")
print(f"  Average:  {np.mean(rank1_gap):.4f}")
print("="*60)

