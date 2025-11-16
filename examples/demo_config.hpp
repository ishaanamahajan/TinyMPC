#pragma once

// Centralized demo configuration for PSD and TV-linear examples.
// Edit these values to change both demos consistently.

// Base dimensions
#define DEMO_NX0 4
#define DEMO_NU0 2

// Horizon
#define DEMO_N 31

// Obstacle (circle) - for default demo only
#define DEMO_OBS_X     (-5.0)
#define DEMO_OBS_Y     (0.0)
#define DEMO_OBS_R     (2.0)
#define DEMO_OBS_MARGIN (0.0)

// Note: Each demo now specifies its own rho values for better tuning

