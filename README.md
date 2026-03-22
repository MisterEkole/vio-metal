# vio-metal

Real-time stereo visual-inertial odometry on Apple Silicon with Metal GPU acceleration.

## Overview

**vio-metal** fuses stereo camera frames and IMU measurements to produce 6-DoF pose estimates in real time, targeting sub-30ms per-frame latency on Apple Silicon. The system exploits the unified memory architecture (UMA) for zero-copy CPU↔GPU data sharing.

The project now features two parallel front-end tracking pipelines: a **CPU fallback version** using OpenCV, and a **hybrid GPU+CPU version** using custom Metal compute shaders for feature detection and stereo matching, with KLT tracking on CPU. Both versions feed into a sliding window optimization backend powered by Ceres Solver.

### Implemented Metal GPU Kernels

The GPU pipeline offloads key vision front-end components to Apple Silicon via custom `.metal` compute shaders:

- **Metal Undistort:** GPU-accelerated stereo image rectification.
- **Metal FastDetect:** FAST corner detection.
- **Metal HarrisResponse:** Sub-pixel corner scoring and non-maximum suppression (Grid NMS).
- **Metal ORBDescriptor:** 256-bit rotated BRIEF descriptor extraction.
- **Metal StereoMatcher:** Epipolar stereo matching using Hamming distance.
- **CPU KLTTracker:** Pyramidal Lucas-Kanade optical flow tracking (moved to CPU for this branch).

## Requirements

- **macOS** ≥ 14.0 (Sonoma)
- **Xcode** ≥ 15.0
- **CMake** ≥ 3.25

### Dependencies

```bash
brew install cmake eigen ceres-solver opencv yaml-cpp
pip install evo  # for trajectory evaluation
```

**Note:** The real-time 3D trajectory visualizer requires Pangolin, which must be built from source.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Dataset

Download the EuRoC MAV Dataset:

```bash
./scripts/download_euroc.sh ./data/euroc
```

## Run

Both pipelines support the following flags:

| Flag | Description |
|------|-------------|
| `--headless` | Run without the Pangolin visualizer window |
| `--quiet` | Suppress per-frame terminal logging |

By default, both the visualizer and terminal logging are active.

### GPU Pipeline (Metal + CPU KLT)

```bash
# With visualizer + terminal logging
./build/vio-metal-gpu <dataset_path> ./build/shaders.metallib

# Headless (no window) + terminal logging
./build/vio-metal-gpu <dataset_path> ./build/shaders.metallib --headless

# Headless + quiet (logging to files only)
./build/vio-metal-gpu <dataset_path> ./build/shaders.metallib --headless --quiet
```

### CPU Pipeline (OpenCV)

```bash
# With visualizer + terminal logging
./build/vio-metal <dataset_path>

# Headless + terminal logging
./build/vio-metal <dataset_path> --headless
```

### Terminal Output

When not `--quiet`, each keyframe prints a status line:

```
[  42] cost: 3634.8 -> 0.0  iter: 5  lm: 34  res: 113  CONV  pos: (0.88, 2.14, 0.95)  err: 0.003m
```

Fields: frame index, initial/final cost, solver iterations, landmark count, residual count, convergence status, estimated position, position error vs ground truth.

### Evaluation

Run the full pipeline + ATE/RPE evaluation with [evo](https://github.com/MichaelGrupp/evo):

```bash
# GPU pipeline (default)
bash eval/evaluate.sh

# CPU pipeline
bash eval/evaluate.sh cpu
```

This runs the pipeline in headless mode, converts ground truth to TUM format, computes ATE/RPE metrics, and generates cost plots.

### Output

- `results/trajectories/estimated_<timestamp>.txt` — TUM-format trajectory
- `results/configs/cost_log_<timestamp>.csv` — per-keyframe optimizer cost log
- `results/configs/cost_plot_<timestamp>.png` — cost evolution plot
- `results/configs/timing_<timestamp>.csv` — per-frame timing breakdown
- `results/configs/ate_<timestamp>.zip` — ATE results (evo)
- `results/configs/rpe_<timestamp>.zip` — RPE results (evo)

All output files are timestamped to prevent overwriting between runs.

## Architecture & Data Flow

### 1. CPU Pipeline Flow (vio-metal)

![CPU Pipeline Flow](img/VIO-Metal-CPU.png)

### 2. Hybrid GPU + CPU Pipeline Flow (vio-metal-gpu)

![Hybrid GPU + CPU Pipeline Flow](img/VIO-Metal-GPU.png)

**Note:** In this branch (klttrackercpu), KLT tracking has been moved back to CPU for performance evaluation and comparison with the full GPU pipeline.



## Results

Full evaluation results comparing both pipelines on EuRoC V1_01_easy are documented in [docs/results.md](docs/results.md).

### Trajectory Accuracy

| Pipeline | ATE RMSE | RPE RMSE | Avg Frame Time |
|----------|----------|----------|----------------|
| **CPU** (OpenCV ORB) | **3.52 m** | **0.28 m** | 60.7 ms |
| **GPU** (Full Metal) | 59.61 m | 1.84 m | 11.9 ms |

The CPU pipeline achieves stable trajectory estimation. The GPU pipeline now uses the full Metal front-end (FAST → Harris → Metal ORB → Metal StereoMatcher) with CPU KLT tracking. After wiring in Metal ORB and Metal StereoMatcher, ATE improved **365x** (from 21,736m to 59.6m) and landmark starvation was eliminated (0 frames with zero landmarks, down from 1,606). The remaining gap is due to the Metal stereo matcher producing ~3.8x fewer matches per keyframe than OpenCV — tuning `MetalStereoConfig` thresholds and increasing `max_keypoints` are the next steps.

### ATE & RPE Comparison

![ATE & RPE Comparison](docs/img/ate_rpe_comparison.png)

### Cost Evolution

![Cost Comparison](docs/img/cost_comparison.png)

### Per-Stage Timing

| Stage | CPU Avg (ms) | GPU Avg (ms) |
|-------|-------------|-------------|
| Undistort | 0.31 | 1.51 |
| Detect | 1.22 | 1.43 |
| Stereo Match | 0.11 | 3.05 |
| Stereo Retrack | — | 3.37 |
| Temporal Track | 0.56 | 0.37 |
| Optimize | 96.56 | 24.49 |
| **Total** | **60.73** | **11.93** |

The GPU pipeline is **5.1x faster** on average. See [full results](docs/results.md) for detailed analysis and path forward.


## Project Structure

```
src/
├── core/           Types, Profiler, KeyframePolicy
├── dataset/        EurocLoader, TrajectoryWriter
├── metal/          MetalContext, MetalUndistort, shaders/ (FAST, ORB, etc.)
├── vision/         FeatureDetector, StereoMatcher, TemporalTracker, FeatureManager
├── imu/            ImuPreintegrator, ImuTypes
├── optimization/   VioOptimizer, Factors (Ceres), Marginalization
├── visualizer.h    Pangolin real-time 3D trajectory plotting
├── main.mm         CPU Pipeline orchestration
└── metal_main.mm   GPU Pipeline orchestration (with CPU KLT tracking)
```

## References

- Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)
- Leutenegger et al. "Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization" (IJRR 2015)
- Qin et al. "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (TRO 2018)
- Burri et al. "The EuRoC Micro Aerial Vehicle Datasets" (IJRR 2016)