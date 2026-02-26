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

The pipeline can be run in either CPU mode (OpenCV vision) or GPU mode (Metal vision with CPU-based KLT tracking). A real-time Pangolin visualizer will spawn to plot the Ground Truth (Red) vs Estimated Trajectory (Green).

**Note:** Replace `<path_to_euroc_dataset>` with the actual path on your local machine.

### Run CPU Version

```bash
./build/vio-metal <path_to_euroc_dataset> ./build/shaders.metallib
```

### Run GPU Version (with CPU-based KLT Tracking)

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

### 2. Hybrid GPU + CPU Pipeline Flow (vio-metal-gpu)

```
Stereo Images ──► Metal Undistort ──► Metal FAST + Harris NMS ──► Metal ORB
                                                                      │
                                          Metal Stereo Matcher ◄──────┤
                                                                      │
                                              CPU KLT Track ◄─────────┤
                                                  │
                                                  ▼
IMU Samples ───► ImuPreintegrator ─────────► Sliding Window Optimizer (Ceres)
                                                    │
                                                    ▼
                                              SE3 Pose Output
```

**Note:** In this branch (klttrackercpu), KLT tracking has been moved back to CPU for performance evaluation and comparison with the full GPU pipeline.



## Performance Benchmarks

The following results were generated on an Apple M-series processor using the EuRoC V1_01_easy dataset (2912 frames).

### Side-by-Side Profiling Summary

| Stage | CPU Only Avg (ms) | Hybrid GPU Avg (ms) | Notes |
|-------|-------------------|-------------------|-------|
| Undistort | 0.27 | 1.06 | GPU includes getBytes sync overhead |
| Detect | 0.31 | 0.18 | GPU Win: FAST + Harris scoring |
| Stereo Match | 0.01 | 0.45 | Hybrid uses ORB descriptor extraction |
| Track | 0.92 | 2.02 | CPU KLT tracking |
| Optimize | 2.52 | 1.35 | GPU Win: Higher quality features |
| Total AVG | 8.99 ms | 9.59 ms | |
| Total MAX | 77.54 ms | 42.78 ms | GPU Win: Drastic reduction in jitter |

### Summary Comparison

**Latency Consistency:** The Hybrid GPU version is significantly more stable. While the CPU version is slightly faster on average, it suffers from massive latency spikes (up to 77ms). The GPU version caps worst-case latency at 42ms, ensuring a much smoother real-time experience.

**Optimization Quality:** The GPU pipeline (Metal FAST + Harris Response) produces higher-quality feature localizations. This is evidenced by the Optimize stage dropping from 2.52ms to 1.35ms, as the backend solver converges much faster with the GPU-sourced data.

**Resource Balancing:** By offloading Undistort and Detection to Metal, the CPU is freed up from feature extraction tasks. 
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
├── main.cpp        CPU Pipeline orchestration
└── metal_main.mm   GPU Pipeline orchestration (with CPU KLT tracking)
```

## References

- Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)
- Leutenegger et al. "Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization" (IJRR 2015)
- Qin et al. "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (TRO 2018)
- Burri et al. "The EuRoC Micro Aerial Vehicle Datasets" (IJRR 2016)
