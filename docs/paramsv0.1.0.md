# VIO-Metal v0.1.0 - Parameter Tuning Log

**Dataset:** EuRoC V1_01_easy (2912 frames, 20Hz stereo + IMU)
**Platform:** Apple M-series, Metal GPU pipeline

This document tracks parameter configurations across tuning trials. Each trial records the full parameter set and observed behavior.

---

## Parameter Reference

### Optimizer (`VioOptimizer::Config` — `src/optimization/VioOptimizer.h`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `window_size` | int | Max keyframes in sliding window before marginalization |
| `max_iterations` | int | Ceres solver max iterations per optimize() call |
| `huber_reprojection` | double | Huber loss threshold (pixels) for reprojection factors |
| `huber_imu` | double | Huber loss threshold for IMU factors (0 = none; hardcoded to 10.0 in optimize()) |
| `use_dogleg` | bool | Use Dogleg trust-region instead of Levenberg-Marquardt |

### Hardcoded Optimizer Parameters (`src/optimization/VioOptimizer.cpp`)

| Parameter | Description |
|-----------|-------------|
| `linear_solver_type` | Ceres linear solver (DENSE_QR / DENSE_SCHUR) |
| `num_threads` | Ceres solver threads |
| `sqrt_info_mono` | Reprojection weight for mono observations (= 1/sigma_pixels) |
| `sqrt_info_stereo` | Reprojection weight for stereo observations |
| `depth_filter_min` | Min depth (m) for landmark visibility check |
| `depth_filter_max` | Max depth (m) for landmark visibility check |
| `huber_imu` (hardcoded) | Huber loss on IMU residual blocks |
| `huber_loop` (hardcoded) | Huber loss on loop closure residual blocks |

### Keyframe Policy (`KeyframePolicy::Config` — `src/core/KeyframePolicy.h`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_tracked_features` | int | Insert KF if tracked features drop below this |
| `min_parallax_deg` | double | Min average parallax (degrees) to trigger KF |
| `min_frames_between` | int | Min frames between keyframes |
| `max_frames_between` | int | Max frames between keyframes (force KF) |

### KLT Tracker (`TemporalTracker::Config` — `src/vision/TemporalTracker.h`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `win_size` | cv::Size | Lucas-Kanade window size |
| `max_level` | int | Pyramid levels |
| `max_iterations` | int | LK solver iterations per level |
| `epsilon` | double | LK convergence threshold |
| `max_error` | double | Max tracking error to accept a point |
| `min_eigen_threshold` | double | Min eigenvalue for trackable corners |
| `use_forward_backward` | bool | Run backward pass for FB consistency check |

### Feature Detection — GPU Pipeline (`src/metal_main.mm`)

| Parameter | Description |
|-----------|-------------|
| `gridNMS` max features (detect) | Max corners after grid NMS for new feature detection |
| `gridNMS` max features (track) | Max corners after grid NMS for tracked frame re-detection |
| `FAST threshold` | Metal FAST detector threshold (set in MetalFastDetector) |

### Feature Detection — CPU Pipeline (`FeatureDetector::Config` — `src/vision/FeatureDetector.h`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_features` | int | Max ORB features to extract |
| `fast_threshold` | int | FAST corner threshold |
| `orb_nlevels` | int | ORB pyramid levels |
| `orb_scale_factor` | float | ORB pyramid scale factor |
| `grid_rows` | int | Grid rows for spatial distribution |
| `grid_cols` | int | Grid columns for spatial distribution |
| `min_distance` | int | Min pixel distance between features |

### Stereo Matcher (`StereoMatcher::Config` — `src/vision/StereoMatcher.h`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_epipolar_error` | double | Max vertical disparity (pixels) after rectification |
| `max_descriptor_dist` | double | Max Hamming distance for ORB match acceptance |
| `min_disparity` | double | Min disparity (pixels) — max depth cutoff |
| `max_disparity` | double | Max disparity (pixels) — min depth cutoff |
| `ratio_test` | double | Lowe's ratio test threshold |

---

## Baseline Configuration

**Date:** 2026-03-21
**Status:** Best working configuration after bug fixes (marginalization e0 sign, extrinsic double inversion)

### Parameters

**Optimizer:**

| Parameter | Value |
|-----------|-------|
| `window_size` | 20 |
| `huber_reprojection` | 1.0 px |
| `linear_solver_type` | DENSE_QR |
| `num_threads` | 8 |
| `sqrt_info_mono` | 1/1.5 (~0.667) |
| `sqrt_info_stereo` | 1/1.5 (~0.667) |
| `depth_filter_min` | 0.1 m |
| `depth_filter_max` | 200.0 m |
| `huber_imu` (hardcoded) | 10.0 |
| `huber_loop` (hardcoded) | 1.0 |

**Keyframe Policy:**

| Parameter | Value |
|-----------|-------|
| `min_tracked_features` | 80 |
| `min_parallax_deg` | 1.0 |
| `min_frames_between` | 2 |
| `max_frames_between` | 10 |

**KLT Tracker:**

| Parameter | Value |
|-----------|-------|
| `win_size` | 11x11 |
| `max_level` | 3 |
| `max_iterations` | 15 |
| `epsilon` | 0.01 |
| `max_error` | 50.0 |
| `min_eigen_threshold` | 1e-4 |
| `use_forward_backward` | false |

**Feature Detection (GPU):**

| Parameter | Value |
|-----------|-------|
| `gridNMS` detect | 400 |
| `gridNMS` track | 300 |
| `FAST threshold` | 20 |

**Feature Detection (CPU):**

| Parameter | Value |
|-----------|-------|
| `max_features` | 500 |
| `fast_threshold` | 20 |
| `orb_nlevels` | 4 |
| `orb_scale_factor` | 1.2 |
| `grid_rows` | 4 |
| `grid_cols` | 5 |
| `min_distance` | 15 |

**Stereo Matcher:**

| Parameter | Value |
|-----------|-------|
| `max_epipolar_error` | 2.0 px |
| `max_descriptor_dist` | 50.0 |
| `min_disparity` | 1.0 px |
| `max_disparity` | 120.0 px |
| `ratio_test` | 0.8 |

### Trial Runs

| Run | `max_iterations` | Run Tag | Date |
|-----|-------------------|---------|------|
| Run A | 100 | `20260322_101351` | 2026-03-22 |
| Run B | 50 | `20260322_104146` | 2026-03-22 |

### Notes

- Configuration established after fixing three critical bugs: marginalization e0 sign, getCorrected() transpose, and double extrinsic inversion
- The reprojection weight `1/1.5` was originally `1/4.5` (a band-aid for the extrinsic bug), restored to `1/1.5` after the extrinsic fix
- DENSE_SCHUR causes Cholesky failures on this problem; DENSE_QR is used as fallback
