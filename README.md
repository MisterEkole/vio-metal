# vio-metal

Real-time stereo visual-inertial odometry on Apple Silicon with Metal GPU acceleration.

## Overview

**vio-metal** fuses stereo camera frames and IMU measurements to produce 6-DoF pose estimates in real time, targeting sub-30ms per-frame latency on Apple Silicon. The system exploits the unified memory architecture (UMA) for zero-copy CPU↔GPU data sharing.

The project now features two parallel front-end tracking pipelines: a **CPU fallback version** using OpenCV, and a **fully accelerated GPU version** using custom Metal compute shaders. Both versions feed into a sliding window optimization backend powered by Ceres Solver.

### Implemented Metal GPU Kernels

The GPU pipeline offloads the entire vision front-end to Apple Silicon via custom `.metal` compute shaders:

- **Metal Undistort:** GPU-accelerated stereo image rectification.
- **Metal FastDetect:** FAST corner detection.
- **Metal HarrisResponse:** Sub-pixel corner scoring and non-maximum suppression (Grid NMS).
- **Metal ORBDescriptor:** 256-bit rotated BRIEF descriptor extraction.
- **Metal StereoMatcher:** Epipolar stereo matching using Hamming distance.
- **Metal KLTTracker:** Pyramidal Lucas-Kanade optical flow tracking.

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

The pipeline can be run in either CPU mode (OpenCV vision) or GPU mode (Metal vision). A real-time Pangolin visualizer will spawn to plot the Ground Truth (Red) vs Estimated Trajectory (Green).

**Note:** Replace `<path_to_euroc_dataset>` with the actual path on your local machine (e.g., `/Users/ekole/Datasets/vicon_room1/V1_01_easy`).

### Run CPU Version

```bash
./build/vio-metal <path_to_euroc_dataset> ./build/shaders.metallib
```

### Run GPU Version

```bash
./build/vio-metal-gpu <path_to_euroc_dataset> ./build/shaders.metallib
```

### Output

- Real-time 3D Pangolin visualization.
- `results/trajectories/estimated.txt` — TUM-format trajectory.
- `results/timing/timing.csv` — per-frame timing breakdown.

## Architecture & Data Flow

### 1. CPU Pipeline Flow (vio-metal)

```
Stereo Images ──► OpenCV Remap (Undistort) ──► OpenCV FAST/ORB Detect
                                                    │
                                                    ├──► OpenCV Stereo Match
                                                    │
                                                    ├──► OpenCV KLT Track
                                                    │
                                                    ▼
IMU Samples ───► ImuPreintegrator ─────────► Sliding Window Optimizer (Ceres)
                                                    │
                                                    ▼
                                              SE3 Pose Output
```

### 2. Metal GPU Pipeline Flow (vio-metal-gpu)

```
Stereo Images ──► Metal Undistort ──► Metal FAST + Harris NMS ──► Metal ORB
                                                                      │
                                          Metal Stereo Matcher ◄──────┤
                                                                      │
                                             Metal KLT Tracker ◄──────┤
                                                                      │
                                                                      ▼
IMU Samples ───► ImuPreintegrator ─────────► Sliding Window Optimizer (Ceres)
                                                    │
                                                    ▼
                                              SE3 Pose Output
```

## Project Structure

```
src/
├── core/           Types, Profiler, KeyframePolicy
├── dataset/        EurocLoader, TrajectoryWriter
├── metal/          MetalContext, MetalUndistort, shaders/ (FAST, ORB, KLT, etc.)
├── vision/         FeatureDetector, StereoMatcher, TemporalTracker, FeatureManager
├── imu/            ImuPreintegrator, ImuTypes
├── optimization/   VioOptimizer, Factors (Ceres), Marginalization
├── visualizer.h    Pangolin real-time 3D trajectory plotting
├── main.cpp        CPU Pipeline orchestration
└── metal_main.mm   GPU Pipeline orchestration
```

## References

- Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)
- Leutenegger et al. "Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization" (IJRR 2015)
- Qin et al. "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (TRO 2018)
- Burri et al. "The EuRoC Micro Aerial Vehicle Datasets" (IJRR 2016)