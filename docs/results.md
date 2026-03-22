# Baseline Configuration Results

Dataset: EuRoC V1_01_easy (full sequence, 2912 frames)
Pipeline: vio-metal GPU (Metal + CPU hybrid)
Config: Baseline (see `paramsv0.1.0.md`)

## Runs

Two runs with the same baseline config, varying only `max_iterations`:

| Parameter | Run A | Run B |
|---|---|---|
| `max_iterations` | 100 | 50 |
| Run tag | `20260322_101351` | `20260322_104146` |

All other parameters identical (see baseline config in `paramsv0.1.0.md`).

## Summary Statistics

| Metric | Run A (iter=100) | Run B (iter=50) |
|---|---|---|
| Keyframes optimized | 2902 | 2902 |
| Converged | 2892/2902 (99.7%) | 2822/2902 (97.2%) |
| Avg initial cost | 2,679,571 | 815,574 |
| Avg final cost | 2,335,470 | 596,944 |
| Avg cost reduction | 12.8% | 26.8% |
| Avg iterations/solve | 25.3 | 23.2 |
| Avg landmarks | 35 | 35 |
| Avg residuals | 763 | 764 |

### Timing

| Stage | Run A (iter=100) | Run B (iter=50) |
|---|---|---|
| Undistort | 1.13 ms | 1.10 ms |
| Detect | 0.17 ms | 0.18 ms |
| Stereo | 0.56 ms | 0.57 ms |
| Track | 1.34 ms | 1.38 ms |
| **Optimize** | **472.6 ms** (max 166,223) | **390.4 ms** (max 104,199) |
| **Total** | **482.4 ms** | **400.3 ms** |

### Trajectory Error (vs ground truth, SE(3) Umeyama alignment)

| Metric | Run A (iter=100) | Run B (iter=50) |
|---|---|---|
| **ATE RMSE** | 36,081 m | 56,137 m |
| ATE mean | 26,393 m | 44,603 m |
| ATE median | 15,481 m | 33,904 m |
| ATE max | 95,176 m | 132,241 m |
| **RPE RMSE** | 1,815 m | 2,568 m |
| RPE mean | 57.5 m | 91.7 m |
| RPE median | 0.021 m | 0.021 m |
| RPE max | 97,119 m | 137,266 m |

## Cost Plots

### Run A — Baseline (max_iterations=100)

![Run A](../results/configs/cost_plot_20260322_101351.png)

### Run B — Baseline (max_iterations=50)

![Run B](../results/configs/cost_plot_20260322_104146.png)

## Analysis

### Iteration budget: 100 vs 50

Run A (iter=100) achieves higher convergence rate (99.7% vs 97.2%) and lower ATE (36km vs 56km RMSE), but at significant compute cost — 472ms vs 390ms per frame on average, with worst-case spikes of 166s vs 104s. The extra iterations help on difficult frames where the solver needs more steps to find a minimum.

However, Run A has **worse cost reduction** (12.8% vs 26.8%). This is counterintuitive: with more iterations available, Run A's initial costs are 3.3x higher than Run B's. The likely explanation is that Run A's solver, having converged more thoroughly on each prior keyframe, produces marginalization priors that are more tightly linearized — when the next keyframe's state deviates from that linearization point, the prior residual starts very high. Run B's looser convergence produces softer priors that accept larger state changes.

### Both runs diverge catastrophically

Despite near-perfect convergence rates, both trajectories diverge to tens of kilometers from ground truth. The RPE median of 0.021m (identical for both) shows that most consecutive frames have reasonable relative motion, but rare catastrophic jumps (RPE max ~100km) destroy global consistency.

This confirms the pattern from earlier experiments: **optimizer convergence does not imply trajectory accuracy**. The solver finds a local minimum of the cost function, but that minimum is far from the true trajectory.

### Root cause remains the same

Both runs average only **35 landmarks** per optimization. With a window of 20 keyframes, this means fewer than 2 landmarks per keyframe on average — severely under-constrained. The problem degenerates into an IMU-dominated system where vision provides insufficient correction.

The front-end (feature detection → tracking → stereo matching → triangulation) is the bottleneck. The optimizer backend is functioning correctly — it converges reliably and reduces cost — but it cannot produce accurate trajectories without sufficient visual constraints.

### Key takeaway

Halving `max_iterations` from 100 to 50:
- Saves ~17% compute (482→400 ms/frame)
- Loses 2.5% convergence (99.7→97.2%)
- Produces worse ATE (56km vs 36km) but better cost reduction ratio
- Does not change the fundamental problem: landmark starvation

Neither iteration budget matters until the front-end delivers more landmarks. Target: 100+ landmarks per optimization window.
