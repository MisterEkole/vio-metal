# vio-metal

Real-time stereo visual-inertial odometry on Apple Silicon with Metal GPU acceleration.

## Overview

**vio-metal** fuses stereo camera frames and IMU measurements to produce 6-DoF pose estimates in real time, targeting sub-30ms per-frame latency on Apple Silicon. The system exploits the unified memory architecture (UMA) for zero-copy CPU↔GPU data sharing.

### Phase 1 (Current)
- End-to-end pipeline with CPU-based vision (OpenCV) and GPU undistortion (Metal)
- Sliding window optimization with Ceres Solver (Accelerate backend)
- IMU preintegration (Forster et al. 2017)
- Per-frame profiling to identify GPU migration targets

### Phase 2+ (Planned)
- Custom Metal compute shaders for feature detection/description
- CoreML + Neural Engine for learned descriptors
- Vision framework for GPU-accelerated tracking

## Requirements

- **macOS** ≥ 14.0 (Sonoma)
- **Xcode** ≥ 15.0
- **CMake** ≥ 3.25

### Dependencies

```bash
brew install cmake eigen ceres-solver opencv yaml-cpp
pip install evo  # for trajectory evaluation
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## Dataset

Download the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):

```bash
./scripts/download_euroc.sh ./data/euroc
```

## Run

```bash
./build/vio-metal ./data/euroc/V1_01_easy [path/to/undistort.metallib]
```

The metallib path is optional — Phase 1 uses OpenCV CPU undistortion by default.

Output:
- `results/trajectories/estimated.txt` — TUM-format trajectory
- `results/timing/timing.csv` — per-frame timing breakdown

## Evaluate

```bash
# Trajectory accuracy (requires evo)
./eval/evaluate.sh ./data/euroc/V1_01_easy results/trajectories/estimated.txt

# Timing visualization
python3 eval/plot_timing.py results/timing/timing.csv
```

## Architecture

```
Stereo Images ──► Metal Undistort ──► Feature Detection (OpenCV ORB)
                                          │
                                          ├──► Stereo Matching
                                          │
                                          ├──► KLT Temporal Tracking
                                          │
                                          ▼
IMU Samples ──► Preintegration ──► Sliding Window Optimizer (Ceres)
                                          │
                                          ▼
                                    SE3 Pose Output
```

## Project Structure

```
src/
├── core/           Types, Profiler, KeyframePolicy
├── dataset/        EurocLoader, TrajectoryWriter
├── metal/          MetalContext, MetalUndistort, shaders/
├── vision/         FeatureDetector, StereoMatcher, TemporalTracker, FeatureManager
├── imu/            ImuPreintegrator, ImuTypes
├── optimization/   VioOptimizer, Factors (Ceres), Marginalization
└── main.cpp        Pipeline orchestration
```

## References

1. Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)
2. Leutenegger et al. "Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization" (IJRR 2015)
3. Qin et al. "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (TRO 2018)
4. Burri et al. "The EuRoC Micro Aerial Vehicle Datasets" (IJRR 2016)
