# VIO-Metal v0.1.0 - Technical Changelog

This document describes all changes made to the VIO-Metal pipeline in v0.1.0. Each section covers a specific bug or feature: what was wrong, what the fix does, and how it works.

---

## 1. Landmarks Were Never Passed to the Optimizer

### Problem

The optimizer had zero visual constraints. `VioOptimizer::optimize()` builds reprojection residuals by iterating over `landmarks_` (a member map of feature_id -> world position). But nothing in either pipeline ever called `optimizer.setLandmarks()`. The method existed, `FeatureManager::getInitializedLandmarksWorld()` existed to transform camera-frame landmarks into world coordinates, but they were never wired together.

The result: the Ceres problem contained only IMU factors. The system was doing pure inertial dead-reckoning, which drifts unboundedly within seconds.

### Fix

Before each call to `optimizer.optimize()`, we now compute world-frame landmarks from the feature manager and pass them in.

**`src/main.mm` and `src/metal_main.mm`** (added before `optimizer.optimize()`):

```cpp
auto lm_world = feature_manager.getInitializedLandmarksWorld(
    optimizer.latestState().position,
    optimizer.latestState().rotation,
    calib.T_cam0_imu);
optimizer.setLandmarks(lm_world);
optimizer.optimize();
```

`getInitializedLandmarksWorld()` transforms each landmark from its local camera frame to world frame:

```
P_body = R_cam_imu^T * (P_camera - t_cam_imu)   // camera→body (inverse of T_cam_imu)
P_world = q_wb * P_body + p_wb                    // body→world
```

`setLandmarks()` inserts new landmarks into the optimizer's map (existing landmarks are kept as-is since Ceres refines them).

### Impact

The optimizer now has 100-250+ landmark constraints per solve. Reprojection factors anchor the trajectory to the 3D structure of the scene.

---

## 2. GPU Pipeline Missing Rectified Intrinsics

### Problem

After `cv::stereoRectify`, the camera intrinsics change (new focal length, new principal point from the projection matrix P1). The CPU pipeline (`main.mm`) updated `calib.intrinsics_left` with the rectified values. The GPU pipeline (`metal_main.mm`) did not.

This meant the GPU pipeline used the original (pre-rectification) intrinsics in all reprojection factors. Even with landmarks wired correctly, the reprojection residuals would compute wrong pixel projections, causing the optimizer to fight against incorrect visual constraints.

### Fix

**`src/metal_main.mm`** (added after `cv::initUndistortRectifyMap`):

```cpp
double fx_rect = P1.at<double>(0,0);
calib.intrinsics_left  = Eigen::Vector4d(
    fx_rect, P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2));
calib.intrinsics_right = Eigen::Vector4d(
    P2.at<double>(0,0), P2.at<double>(1,1), P2.at<double>(0,2), P2.at<double>(1,2));
double baseline_rect = -P2.at<double>(0,3) / fx_rect;
calib.T_cam1_cam0 = Eigen::Matrix4d::Identity();
calib.T_cam1_cam0(0,3) = -baseline_rect;
```

This matches what the CPU pipeline already had.

---

## 3. Bias Jacobian Update-Order Bug in IMU Preintegration

### Problem

In `ImuPreintegrator::integrate()`, the bias Jacobians (`J_bg_`, `J_ba_`) track how the preintegrated measurements change when the gyro/accel biases deviate from their linearization point. These Jacobians have three rows: rotation (0:3), velocity (3:6), and position (6:9).

The original code updated them in-place, top to bottom:

```cpp
// WRONG: uses already-updated J_bg_(0:3) in the velocity update
J_bg_.block<3,3>(0,0) = dR_inc.transpose() * J_bg_.block<3,3>(0,0) - Jr * dt;
J_bg_.block<3,3>(3,0) = J_bg_.block<3,3>(3,0)
    - R_k * skewSymmetric(acc) * J_bg_.block<3,3>(0,0) * dt;  // <-- uses NEW rotation Jacobian
J_bg_.block<3,3>(6,0) = J_bg_.block<3,3>(6,0) + J_bg_.block<3,3>(3,0) * dt
    - 0.5 * R_k * skewSymmetric(acc) * J_bg_.block<3,3>(0,0) * dt * dt;  // <-- uses NEW values
```

The velocity Jacobian update at step `k` depends on the rotation Jacobian at step `k-1`, but it was reading the already-overwritten step `k` value. Same issue cascaded to the position Jacobian.

### Fix

**`src/imu/ImuPreintegrator.cpp`** - save old values before overwriting:

```cpp
Eigen::Matrix3d J_bg_R_old = J_bg_.block<3,3>(0,0);
Eigen::Matrix3d J_bg_v_old = J_bg_.block<3,3>(3,0);
Eigen::Matrix3d J_ba_v_old = J_ba_.block<3,3>(3,0);

J_bg_.block<3,3>(0,0) = dR_inc.transpose() * J_bg_R_old - Jr * dt;
J_bg_.block<3,3>(3,0) = J_bg_v_old - R_k * skewSymmetric(acc) * J_bg_R_old * dt;
J_bg_.block<3,3>(6,0) = J_bg_.block<3,3>(6,0) + J_bg_v_old * dt
    - 0.5 * R_k * skewSymmetric(acc) * J_bg_R_old * dt * dt;

J_ba_.block<3,3>(3,0) = J_ba_v_old - R_k * dt;
J_ba_.block<3,3>(6,0) = J_ba_.block<3,3>(6,0) + J_ba_v_old * dt - 0.5 * R_k * dt * dt;
```

The discrete-time Jacobian update equations (from Forster et al. 2017, "On-Manifold Preintegration") are:

```
d(delta_R)/d(b_g)[k+1] = dR_inc^T * d(delta_R)/d(b_g)[k] - Jr * dt

d(delta_v)/d(b_g)[k+1] = d(delta_v)/d(b_g)[k]
    - R_k * [acc]_x * d(delta_R)/d(b_g)[k] * dt

d(delta_p)/d(b_g)[k+1] = d(delta_p)/d(b_g)[k]
    + d(delta_v)/d(b_g)[k] * dt
    - 0.5 * R_k * [acc]_x * d(delta_R)/d(b_g)[k] * dt^2
```

All right-hand-side references must use step `k` values, not step `k+1`.

---

## 4. Schur Complement Marginalization

### Problem

The original `VioOptimizer::marginalize()` simply popped the oldest keyframe from the window and discarded all its factors. Later, a diagonal prior was added as a temporary fix:

```cpp
H.block<3,3>(0,0)  = I * 1e2;   // position
H.block<3,3>(3,3)  = I * 1e2;   // rotation
H.block<3,3>(6,6)  = I * 1e1;   // velocity
H.block<3,3>(9,9)  = I * 1e3;   // gyro bias
H.block<3,3>(12,12) = I * 1e3;  // accel bias
```

This is numerically stable but carries no actual information from the removed factors. The IMU constraint between the marginalized frame and its neighbor, and all reprojection factors from that frame, are silently discarded. Over time this causes information loss and drift.

### Fix: Proper Schur Complement

The new `marginalize()` builds a mini Ceres problem containing only the factors touching the oldest keyframe (frame 0), evaluates the full Jacobian, and applies the Schur complement to produce an information-preserving prior on the boundary frame (frame 1).

**Algorithm:**

1. **Build mini problem** with:
   - The existing marginalization prior on frame 0 (if any, from the previous marginalization)
   - The IMU factor between frame 0 and frame 1
   - All reprojection factors from frame 0 to its observed landmarks

2. **Partition variables** into:
   - **Marginalize (m):** frame 0 state (15 tangent dims) + all landmarks observed only by frame 0 (3 dims each)
   - **Keep (k):** frame 1 state (15 tangent dims)

3. **Evaluate Jacobians** via `ceres::Problem::Evaluate()`, which returns a sparse (CRS) Jacobian matrix `J` and residual vector `r`.

4. **Compute the Hessian and gradient:**

```
H = J^T * J
b = -J^T * r
```

5. **Partition the Hessian** into blocks corresponding to marginalized and kept variables:

```
    [ H_mm  H_mk ]       [ b_m ]
H = [             ]   b = [     ]
    [ H_km  H_kk ]       [ b_k ]
```

6. **Schur complement:**

```
H_prior = H_kk - H_km * H_mm^{-1} * H_mk
b_prior = b_k  - H_km * H_mm^{-1} * b_m
```

This is the exact marginalization formula. `H_prior` is the information matrix and `b_prior` is the gradient of the prior cost on the kept variables.

7. **Decompose into sqrt form** for Ceres. Eigendecomposition of `H_prior`:

```
H_prior = V * D * V^T
sqrt_info = D^{1/2} * V^T
e0 = D^{-1/2} * V^T * b_prior
```

The prior cost function evaluates:

```
residual = sqrt_info * (x - x0) + e0
```

where `x0` is the linearization point (frame 1's state at marginalization time).

**`src/optimization/Marginalization.cpp`** - `setPrior()`:

```cpp
Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_reg);
Eigen::VectorXd eigenvalues = solver.eigenvalues();
Eigen::MatrixXd eigenvectors = solver.eigenvectors();

for (int i = 0; i < dim_; i++) {
    if (eigenvalues(i) > 1e-6) {
        sqrt_eigenvalues(i) = std::sqrt(eigenvalues(i));
        inv_sqrt_eigenvalues(i) = 1.0 / sqrt_eigenvalues(i);
    } else {
        sqrt_eigenvalues(i) = 0.0;
        inv_sqrt_eigenvalues(i) = 0.0;
    }
}

sqrt_info_ = sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose();
e0_ = inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
```

**`src/optimization/Marginalization.h`** - `MarginalizationFactor`:

```cpp
bool operator()(const double* const p, const double* const q,
                const double* const vba, double* residuals) const {
    // ... compute tangent-space delta dx from linearization point x0 ...
    res = sqrt_info_ * dx + e0_;
    return true;
}
```

### Pointer Stability

The mini problem stores landmark parameters in a `std::vector<std::array<double,3>>`. To prevent iterator/pointer invalidation during `push_back`, the vector is pre-counted and `reserve()`d before any elements are added or registered with Ceres.

### Fallback

If `ceres::Problem::Evaluate()` fails or the Jacobian dimensions don't match, the system falls back to the diagonal prior.

---

## 5. Loop Closure

### Problem

A sliding-window optimizer discards old keyframes during marginalization. Even with a proper Schur complement, information is gradually lost. If the camera revisits a previously seen area, there is no mechanism to recognize this and correct accumulated drift.

### Solution

A new `LoopDetector` class maintains a database of keyframe descriptors. When a new keyframe is created, it is compared against past (non-recent) keyframes using brute-force ORB matching. If a match passes geometric verification via PnP, a relative pose constraint is added to the optimizer.

### Architecture

```
New Keyframe
    |
    v
LoopDetector::addKeyframe()     <-- stores descriptors + landmarks
    |
    v
LoopDetector::detectLoop()
    |
    +-- BFMatcher (Hamming) + Lowe's ratio test
    |
    +-- Build 3D-2D correspondences (candidate landmarks -> query keypoints)
    |
    +-- cv::solvePnPRansac()     <-- geometric verification
    |
    +-- Recover body-frame pose via T_cam_imu
    |
    v
LoopConstraint --> VioOptimizer::addLoopConstraint()
    |
    v
Re-optimize with loop factor
```

### New Files

**`src/vision/LoopDetector.h`** - Data structures and interface:

```cpp
struct KeyframeDescriptorEntry {
    uint64_t timestamp;
    cv::Mat descriptors;                                  // ORB (N x 32, CV_8U)
    std::vector<cv::KeyPoint> keypoints;
    std::vector<uint64_t> feature_ids;
    std::unordered_map<uint64_t, Eigen::Vector3d> landmarks_world;  // snapshot at insertion
};

struct LoopCandidate {
    uint64_t query_timestamp;
    uint64_t match_timestamp;
    Eigen::Vector3d relative_position;      // query body pose in world frame (from PnP)
    Eigen::Quaterniond relative_rotation;
    int num_inliers;
    bool valid;
};
```

**`src/vision/LoopDetector.cpp`** - Detection pipeline:

1. **Descriptor matching:** `cv::BFMatcher(NORM_HAMMING)` with `knnMatch(k=2)` and Lowe's ratio test (threshold 0.75). Only candidates with >= 25 good matches are considered.

2. **3D-2D correspondence building:** For each match, the 3D point comes from the candidate keyframe's stored `landmarks_world` (looked up by feature_id). The 2D point comes from the query keyframe's keypoint.

3. **PnP verification:** `cv::solvePnPRansac` with 200 iterations and 5px reprojection threshold. Requires >= 12 inliers.

4. **Body-frame pose recovery:** PnP gives `T_camera_world` for the query. Convert to body frame:

```
R_wb = R_wc * R_ci
t_wb = R_wc * t_ci + t_wc
```

where `R_ci, t_ci` are from `T_cam_imu` (camera-to-IMU extrinsic).

### Loop Closure Factor

**`src/optimization/Factors.h`** - `LoopClosureFactor`:

A relative pose factor with 6 residuals (3 translation + 3 rotation). Given two keyframe poses `(p_i, q_i)` and `(p_j, q_j)`, and a measured relative transform `(rel_p, rel_q)`:

```
dp_computed = q_i^{-1} * (p_j - p_i)
dq_computed = q_i^{-1} * q_j

r_position = dp_computed - rel_p_measured
r_rotation = 2 * vec( rel_q_measured^{-1} * dq_computed )

residual = sqrt_info * [r_position; r_rotation]
```

The `sqrt_info` is set to `10 * I_6` (moderate confidence).

```cpp
template <typename T>
bool operator()(const T* const pi, const T* const qi,
                const T* const pj, const T* const qj,
                T* residuals) const {
    // ...
    Eigen::Matrix<T, 3, 1> dp = q_i.inverse() * (p_j - p_i);
    Eigen::Quaternion<T> dq = q_i.inverse() * q_j;

    res.template segment<3>(0) = dp - rel_p_.template cast<T>();

    Eigen::Quaternion<T> dq_err = rel_q_.template cast<T>().inverse() * dq;
    if (dq_err.w() < T(0)) { dq_err.coeffs() *= T(-1); }
    res.template segment<3>(3) = T(2.0) * dq_err.vec();

    res = sqrt_info_.template cast<T>() * res;
    return true;
}
```

Uses `ceres::AutoDiffCostFunction<LoopClosureFactor, 6, 3, 4, 3, 4>`.

### Optimizer Integration

**`src/optimization/VioOptimizer.h`** - new members:

```cpp
struct LoopConstraint {
    uint64_t timestamp_i;               // candidate (match) keyframe
    uint64_t timestamp_j;               // query keyframe
    Eigen::Vector3d relative_position;  // T_ij translation in frame i
    Eigen::Quaterniond relative_rotation;
    Eigen::Matrix<double, 6, 6> sqrt_info;
};

std::vector<LoopConstraint> loop_constraints_;
void addLoopConstraint(const LoopConstraint& lc);
```

**`src/optimization/VioOptimizer.cpp`** - in `optimize()`, after IMU residuals:

```cpp
for (const auto& lc : loop_constraints_) {
    int idx_i = -1, idx_j = -1;
    for (int i = 0; i < n; ++i) {
        if (window_[i].timestamp == lc.timestamp_i) idx_i = i;
        if (window_[i].timestamp == lc.timestamp_j) idx_j = i;
    }
    if (idx_i >= 0 && idx_j >= 0) {
        auto* cf = new ceres::AutoDiffCostFunction<LoopClosureFactor, 6, 3, 4, 3, 4>(
            new LoopClosureFactor(lc.relative_position, lc.relative_rotation, lc.sqrt_info));
        problem.AddResidualBlock(cf, new ceres::HuberLoss(1.0),
            params[idx_i].p, params[idx_i].q, params[idx_j].p, params[idx_j].q);
    }
}
```

Stale loop constraints (referencing marginalized keyframes) are pruned in `marginalize()`.

### Pipeline Integration

In both `main.mm` and `metal_main.mm`, after each keyframe optimization:

1. Build a `KeyframeDescriptorEntry` with the current frame's ORB descriptors, keypoints, feature IDs, and world-frame landmarks.
2. Add it to the loop detector database.
3. Query `detectLoop()` against the database.
4. If a valid loop is detected:
   - Look up the candidate's pose from the optimizer window.
   - Compute the relative transform: `T_ij = T_i^{-1} * T_j`.
   - Add as a `LoopConstraint`.
   - Re-optimize.

---

## 6. KLT Tracking Parameter Optimization

### Problem

The temporal tracker (`TemporalTracker`) was the second most expensive pipeline stage at 2.08 ms average (after the optimizer). Profiling showed:

```
Stage           | Avg (ms) | Max (ms)
----------------|----------|--------
Track           | 2.08     | 13.35
```

The parameters were configured for maximum robustness at the expense of speed:

```cpp
// Old defaults
cv::Size win_size = cv::Size(21, 21);  // 441 pixels per window
int max_iterations = 30;
bool forward_backward = true;          // implicit — always ran 2 full KLT passes
```

The forward-backward consistency check doubles the cost: it runs a second full `cv::calcOpticalFlowPyrLK` call from curr->prev, then rejects any point where the round-trip error exceeds 1 pixel.

A separate project (vx-rs) uses the same Metal KLT shader kernel but with different parameters — 11x11 window, 15 iterations, no backward pass — and runs significantly faster.

### Analysis

For EuRoC at 20Hz, inter-frame displacements are small (a few pixels). A 21x21 window is larger than necessary. The 4-level pyramid handles coarse alignment at upper levels; the fine level only needs to refine by sub-pixel amounts, which converges in fewer iterations.

The forward-backward check is valuable as a standalone outlier filter, but in this pipeline it is redundant: stereo matching provides independent 3D constraints, and the Huber loss in the optimizer already downweights reprojection outliers.

**Cost breakdown per tracked point (500 points, 4 pyramid levels):**

| Config | Pixels/window | Iterations | Passes | Total MACs |
|--------|--------------|------------|--------|------------|
| Old (21x21, FB on) | 441 | 30 | 2 | 441 x 30 x 2 x 4 = 105,840 |
| New (11x11, FB off) | 121 | 15 | 1 | 121 x 15 x 1 x 4 = 7,260 |
| **Reduction** | | | | **~14.6x** |

### Fix

**`src/vision/TemporalTracker.h`** — updated defaults:

```cpp
struct Config {
    cv::Size win_size = cv::Size(11, 11);      // was 21x21
    int max_level = 3;
    int max_iterations = 15;                    // was 30
    double epsilon = 0.01;
    double max_error = 50.0;
    double min_eigen_threshold = 1e-4;
    bool use_forward_backward = false;          // new flag, default off
};
```

**`src/vision/TemporalTracker.mm`** — the backward pass is now conditional:

```cpp
cv::calcOpticalFlowPyrLK(prev_image, curr_image,
    prev_points, fwd_pts, fwd_status, fwd_err,
    config_.win_size, config_.max_level, criteria,
    0, config_.min_eigen_threshold);

if (config_.use_forward_backward) {
    // Run backward pass and apply FB consistency check (1px threshold)
    cv::calcOpticalFlowPyrLK(curr_image, prev_image,
        fwd_pts, bwd_pts, bwd_status, bwd_err, ...);
    // ... validate with fb_err > 1.0 rejection ...
} else {
    // Forward-only: eigenvalue + error threshold + bounds check
    for (size_t i = 0; i < prev_points.size(); i++) {
        if (!fwd_status[i]) continue;
        if (fwd_err[i] > config_.max_error) continue;
        if (out of bounds) continue;
        result.tracked_points[i] = fwd_pts[i];
        result.status[i] = true;
        result.num_tracked++;
    }
}
```

The forward-backward check can be re-enabled by constructing `TemporalTracker` with `config.use_forward_backward = true` if needed for datasets with more aggressive motion.

---

## 7. Marginalization Prior Gradient Sign Fix

### Problem

The marginalization prior encodes information from removed keyframes as a cost function:

```
residual = sqrt_info * (x - x0) + e0
cost = 0.5 * ||residual||^2
```

The residual offset `e0` must be set so that the gradient of the prior cost at the linearization point matches the gradient from the Schur complement. The code computed:

```cpp
e0_ = inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
```

where `b = -J^T * r` (the negative gradient from the normal equations). This gives `sqrt_info^T * e0 = b`, meaning the prior's gradient equals `b`. But `b` uses the convention `b = -gradient`, so the prior was pushing the optimizer **away from** the correct minimum instead of toward it.

### Fix

**`src/optimization/Marginalization.cpp:43`:**

```cpp
// BEFORE (wrong — inverted gradient direction):
e0_ = inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;

// AFTER (correct — negative sign ensures prior attracts toward linearization point):
e0_ = -(inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b).eval();
```

The `.eval()` is required to avoid an Eigen expression template issue with the unary negation operator.

### Impact

Convergence rate improved from ~40% to ~94%. The marginalization prior now correctly constrains the boundary keyframe, preventing the optimizer from drifting away from information encoded by removed keyframes.

---

## 8. Double Extrinsic Inversion Bug (T_cam0_imu)

### Problem

The extrinsic transform `T_cam0_imu` is loaded as `T_imu_cam0.inverse()`, meaning it transforms **from IMU to camera**: `P_cam = R_ci * P_imu + t_ci`.

Three locations in the codebase applied this transform in the wrong direction:

**1. `FeatureManager::getInitializedLandmarksWorld` (camera→body):**

```cpp
// WRONG: applies IMU→camera forward to a camera-frame point
P_body = R_ci * landmark_3d + t_ci;
```

Should use the inverse: `P_body = R_ci^T * (P_cam - t_ci)`.

**2. Reprojection factors in `Factors.h` (body→camera):**

```cpp
// WRONG: inverts T_cam0_imu (which is already body→camera) then applies the inverse
R_cb = R_bc.transpose();
t_cb = -R_cb * t_bc;
p_c = R_cb * p_b + t_cb;
```

Should use the forward transform directly: `p_c = R_bc * p_b + t_bc`.

**3. Depth check in `VioOptimizer::optimize`:** Same inverted transform as the factors.

### Why it appeared to work

The two bugs create a double inversion that cancels for the originating keyframe:

```
Landmark init: P_world = q_wb * (R_ci * P_cam + t_ci) + p_wb      (wrong body)
Factor:        p_c = R_ci^T * (q_wb^{-1} * (P_world - p_wb)) - R_ci^T * t_ci
             = R_ci^T * (R_ci * P_cam + t_ci) - R_ci^T * t_ci
             = P_cam                                                 (cancels!)
```

For the keyframe where the landmark was triangulated, the errors cancel perfectly. But for any other keyframe observing the same landmark, the wrong world position produces reprojection errors that grow with the inter-keyframe baseline.

### Observed behavior (before fix)

- KF 1-10 (frames 11-100): Sub-2cm error — landmarks mostly observed from similar viewpoints
- KF 11 (frame ~111): Initial cost jumped from ~200 to ~11000 — camera has moved enough to expose cross-keyframe reprojection errors
- KF 11+: Optimizer degraded good IMU predictions (0.02m → 0.08m) because 3000+ wrong reprojection residuals overpowered ~150 correct IMU residuals
- Result: Positive feedback loop — bad optimizer output → worse next IMU prediction → divergence

### Fix

**`src/vision/FeatureManager.cpp:110-116`:**

```cpp
// Correct camera→body using inverse of T_cam_imu
Eigen::Matrix3d R_ic = R_ci.transpose();
Eigen::Vector3d t_ic = -R_ci.transpose() * t_ci;
P_body = R_ic * track.landmark_3d + t_ic;
```

**`src/optimization/Factors.h` (both MonoReprojectionFactor and StereoReprojectionFactor):**

```cpp
// Correct body→camera using T_cam0_imu directly (no inversion)
Eigen::Matrix<T, 3, 1> p_c = R_bc * p_b + t_bc;
```

**`src/optimization/VioOptimizer.cpp` (depth check):**

```cpp
Eigen::Vector3d p_c = R_bc * p_b + t_bc;
```

### Impact

After fix, the optimizer maintains sub-3cm error through KF 11 (previously diverging at multi-meter scale). Initial cost at KF 10 dropped from ~11000 to ~400. The optimizer now improves IMU predictions rather than degrading them.

---

## 9. Metal Command Buffer Deadlock in GPU Pipeline

### Problem

The GPU pipeline (`metal_main.mm`) hung indefinitely after processing the first frame. Process sampling (`sample <pid>`) confirmed 100% of CPU time was spent in `MetalContext::waitForLastBuffer()` → `[_MTLCommandBuffer waitUntilCompleted]`, waiting on a Metal command buffer that never completed.

The root cause was a use-after-free in the deferred command buffer pattern used by `MetalUndistort`. The pipeline used a shared `last_buffer_` pointer in `MetalContext` to defer GPU synchronization:

**`src/metal/MetalUndistort.mm` (old):**

```objc
[commandBuffer commit];
context_->setLastBuffer((__bridge void*)commandBuffer);
```

**`src/metal/MetalContext.mm`:**

```cpp
void MetalContext::setLastBuffer(void* buf) { last_buffer_ = buf; }

void MetalContext::waitForLastBuffer() {
    if (!last_buffer_) return;
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)last_buffer_;
    [buf waitUntilCompleted];
    last_buffer_ = nullptr;
}
```

The `(__bridge void*)` cast stores the raw pointer **without retaining** the Objective-C object. Under ARC, the `commandBuffer` local variable inside `encodeUndistort()` is the last strong reference. When `encodeUndistort()` returns, ARC inserts a release. At that point the only thing keeping the command buffer alive is Metal's internal reference to committed-but-incomplete buffers — but this is an implementation detail, not a guaranteed lifetime.

On subsequent frames, the autorelease pool could recycle the memory. The `last_buffer_` pointer would then reference either:
1. A deallocated object → `waitUntilCompleted` sends a message to a zombie (hang or crash)
2. A recycled object at the same address → `waitUntilCompleted` waits on an unrelated object that may never complete

A compounding issue: the `vio_loop` lambda wrapped its entire body (including the `while` loop) in a single `@autoreleasepool`. Autoreleased Metal objects from every iteration accumulated without draining, increasing memory pressure and the likelihood of the recycled-pointer scenario.

Meanwhile, `MetalFastDetector` and `MetalHarrisResponse` used a different, self-contained pattern that was immune to this bug:

**`src/metal/FastDetect.mm` (already correct):**

```objc
[commandBuffer commit];
[commandBuffer waitUntilCompleted];  // synchronous — buffer is alive (local strong ref)
```

These classes never called `setLastBuffer()`, so `last_buffer_` retained a stale pointer from the previous `encodeUndistort()` call. The `waitForLastBuffer()` calls after FAST/Harris detection were no-ops (waiting on an already-completed or dangling buffer), masking the issue during the init frame but exposing it on frame 2+.

### Why it appeared to work on frame 0

On the first iteration (init), the sequence was:

```
encodeUndistort(left)  → sets last_buffer_
waitForLastBuffer()    → waits, clears last_buffer_  ✓
encodeUndistort(right) → sets last_buffer_
waitForLastBuffer()    → waits, clears last_buffer_  ✓
fast->detect()         → self-contained wait          ✓ (last_buffer_ is nullptr)
harris->score()        → self-contained wait          ✓ (last_buffer_ is nullptr)
```

Each `waitForLastBuffer()` ran immediately after `encodeUndistort()` in the same scope, before ARC could release the command buffer. On frame 2, the same pattern ran — but with accumulated autorelease pool pressure and potential pointer recycling from frame 1's objects, the buffer referenced by `last_buffer_` was no longer valid.

### Fix

Two changes eliminate the bug:

**1. Make `MetalUndistort` synchronous (like FAST/Harris):**

**`src/metal/MetalUndistort.mm` (new):**

```objc
[commandBuffer commit];
[commandBuffer waitUntilCompleted];  // wait while local strong reference is alive
```

The command buffer is now waited on while the local `commandBuffer` variable still holds a strong ARC reference, guaranteeing the object is alive. The fragile `setLastBuffer()` / `waitForLastBuffer()` round-trip through an unretained `void*` is eliminated entirely.

This has no performance impact because `metal_main.mm` always called `waitForLastBuffer()` immediately after `encodeUndistort()` — there was never any actual async overlap.

**2. Add per-iteration `@autoreleasepool` in the frame loop:**

**`src/metal_main.mm` (new):**

```objc
while (loader.hasNext() && ...) {
    @autoreleasepool {
        // ... entire frame processing ...
    } // drains all autoreleased Metal objects each iteration
}
```

Previously the autorelease pool wrapped the entire lambda body, meaning all autoreleased Metal command buffers, textures, and encoder objects from every frame accumulated until the pipeline finished. With 2912 frames at ~3 Metal dispatches each, this could accumulate ~8700 autoreleased objects before a single drain.

The per-iteration pool ensures Metal objects are released promptly after each frame, preventing unbounded memory growth and eliminating stale pointer reuse.

**3. Removed stale `waitForLastBuffer()` calls from `metal_main.mm`:**

Since `encodeUndistort()` now waits internally, and `FastDetect`/`HarrisResponse` always waited internally, all `metal_ctx->waitForLastBuffer()` calls in the pipeline were redundant and have been removed.

### The `__bridge` vs `__bridge_retained` distinction

This bug is a classic ARC bridging pitfall. The three Objective-C bridge casts behave differently:

| Cast | Retain count change | Use case |
|------|-------------------|----------|
| `__bridge` | No change | Temporary peek at the pointer; object must be kept alive by another strong reference |
| `__bridge_retained` | +1 (caller must `CFRelease`) | Transferring ownership to a `void*` that outlives the current scope |
| `__bridge_transfer` | -1 (ARC takes ownership back) | Reclaiming a `void*` that was previously `__bridge_retained` |

The original code used `__bridge` to store into `last_buffer_` — a `void*` that outlived the scope — without retaining. An alternative fix would have been to use `__bridge_retained` in `setLastBuffer` and `__bridge_transfer` in `waitForLastBuffer`, but making the operation synchronous is simpler and equally performant for this use case.

### Impact

The GPU pipeline now processes all 2912 frames of V1_01_easy without hanging. Landmarks grow from 19 → 168+ as stereo re-matching kicks in. Position error stays at 1-2cm through the first 30+ keyframes, matching the CPU pipeline's accuracy.

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `src/imu/ImuPreintegrator.cpp` | Fixed bias Jacobian update ordering; removed spurious `.transpose()` in `getCorrected()` |
| `src/optimization/Marginalization.h` | Rewrote `MarginalizationFactor` to accept and apply gradient offset `e0` |
| `src/optimization/Marginalization.cpp` | `setPrior()` now computes both `sqrt_info` and `e0` from eigendecomposition; **fixed e0 sign** |
| `src/optimization/VioOptimizer.h` | Added `LoopConstraint`, `loop_constraints_`, `addLoopConstraint()` |
| `src/optimization/VioOptimizer.cpp` | Rewrote `marginalize()` with Schur complement; added loop closure residuals in `optimize()`; **fixed body→camera depth check** |
| `src/optimization/Factors.h` | Added `LoopClosureFactor` (6D relative pose); **fixed body→camera transform in mono and stereo reprojection factors** |
| `src/vision/FeatureManager.cpp` | **Fixed camera→body transform** in `getInitializedLandmarksWorld()` (was applying forward instead of inverse) |
| `src/vision/LoopDetector.h` | New file: loop detector interface and data structures |
| `src/vision/LoopDetector.cpp` | New file: ORB matching, PnP verification, body-frame pose recovery |
| `src/vision/TemporalTracker.h` | Window 21x21 -> 11x11, iterations 30 -> 15, added `use_forward_backward` flag (default off) |
| `src/vision/TemporalTracker.mm` | Forward-backward pass now conditional on config flag |
| `src/main.mm` | Wired landmarks to optimizer; integrated loop detector; added `--headless`/`--quiet` flags |
| `src/metal/MetalUndistort.mm` | **Fixed command buffer deadlock**: replaced deferred `setLastBuffer()` with synchronous `waitUntilCompleted`; removed stale `waitForLastBuffer()` from `undistort()` convenience method |
| `src/metal_main.mm` | Wired landmarks to optimizer; added rectified intrinsic update; integrated loop detector; added `--headless`/`--quiet` flags; **removed stale `waitForLastBuffer()` calls; added per-iteration `@autoreleasepool`** |
| `src/dataset/TrajectoryWriter.cpp` | Removed comment header line that broke evo TUM parser |
| `eval/evaluate.sh` | Added `cpu`/`gpu` mode selection; GT-to-TUM conversion for evo compatibility |
| `eval/plot_cost.py` | Made `plt.show()` conditional to avoid blocking in headless mode |
| `CMakeLists.txt` | Added `src/vision/LoopDetector.cpp` to `VIO_SOURCES` |

---

# v0.2.0 - Fixes After v0.1.0

These changes reduced the final position error on EuRoC V1_01_easy from ~94 km to ~4.3 m (a 21,000x improvement).

---

## 10. Wrong Distortion Model for Stereo Rectification

### Problem

EuRoC cameras use the **equidistant (fisheye)** distortion model with 4 parameters (k1, k2, k3, k4). The pipeline was calling OpenCV's standard pinhole functions:

```cpp
cv::stereoRectify(K_left, dist_left, K_right, dist_right, ...);
cv::initUndistortRectifyMap(K_left, dist_left, R1, P1, ...);
```

These functions assume a radial-tangential distortion model. Applying them to fisheye coefficients produces:
- **Incorrect undistortion maps** — pixels are remapped to wrong locations, so rectified images are geometrically distorted.
- **Wrong projection matrix P1** — the resulting focal length (fx=436) was neither the raw value (458) nor the correct rectified value (369). Every reprojection in the optimizer used the wrong intrinsics.
- **Wrong rectification rotations R1/R2** — the epipolar alignment was approximate, degrading stereo matching and triangulation accuracy.

This was the **root cause** of the trajectory divergence. With wrong intrinsics and distorted images, the reprojection errors grew monotonically and the optimizer could not converge to a correct solution.

### Fix

Switched both pipelines to OpenCV's fisheye module:

**`src/main.mm`:**
```cpp
cv::fisheye::stereoRectify(K_left, dist_left, K_right, dist_right,
                  cv::Size(752, 480), R_rl_cv, t_rl_cv,
                  R1, R2, P1, P2, Q,
                  cv::CALIB_ZERO_DISPARITY, cv::Size(752, 480));

cv::fisheye::initUndistortRectifyMap(K_left, dist_left, R1, P1,
                  cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
cv::fisheye::initUndistortRectifyMap(K_right, dist_right, R2, P2,
                  cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);
```

Same change in **`src/metal_main.mm`**.

After the fix, P1 reports `fx=368.651` (correct for the fisheye→pinhole rectification with `balance=0`), and self-reprojection errors at the anchor keyframe drop to ~0.1 px.

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Rectified fx | 436.2 (wrong) | 368.7 (correct) |
| Self-reprojection error (KF0) | ~1.6 px | ~0.1 px |
| Cross-frame reproj error (2 KF) | ~1.6 px avg | ~0.11 px avg |

---

## 11. First Keyframe Observations Missing from Optimizer

### Problem

At initialization, the pipeline detected features and stereo-matched them at the first frame (KF0), but **never registered KF0's pixel observations with the optimizer**. The call sequence was:

```cpp
optimizer.initialize(init_state);                    // creates KF0 in the window
feature_manager.addNewFeatures(ts, kpts, desc, sm);  // triangulates landmarks
// ... missing: no addObservations / addKeyframe for KF0
```

When the second keyframe (KF1) was added and `optimize()` ran, the optimizer had:
- KF0 fixed as gauge (correct pose from ground truth)
- KF0 had **zero observations** in `observations_` → no vision residuals anchored to KF0
- Landmarks triangulated at KF0 were only constrained by their observations at KF1+

Without KF0's observations, the fundamental triangulation constraint was lost. Landmark positions floated freely relative to the fixed anchor, and subsequent optimizations started from weak initial conditions that snowballed into drift.

### Fix

After feature detection at initialization, store KF0's observations and initial landmark world positions in the optimizer:

**`src/main.mm`:**
```cpp
feature_manager.addNewFeatures(frame.timestamp_ns, det.keypoints, det.descriptors, stereo_matches);

// NEW: register KF0's observations and landmarks
optimizer.addObservations(frame.timestamp_ns,
    feature_manager.getObservationsForFrame(frame.timestamp_ns));
auto lm_world_init = feature_manager.getInitializedLandmarksWorld(
    init_state.position, init_state.rotation, calib.T_cam0_imu);
optimizer.setLandmarks(lm_world_init);
```

Same change in **`src/metal_main.mm`**.

### Impact

At the first 2-keyframe optimization, KF0 now contributes ~130 vision residuals (avg error 0.10 px), properly anchoring all landmark positions to the ground-truth initial pose.

---

## 12. Position Jump Rejection Caused a Death Spiral

### Problem

The optimizer had a safety check that rejected solutions where the latest keyframe position moved more than 2 m from the IMU prediction:

```cpp
if (jump > config_.max_position_jump) {
    // Reject: keep IMU prediction, reset marginalization prior
    return window_.back();
}
```

This created a **feedback loop**:
1. As the sliding window moves, the fixed-oldest-frame gauge causes natural position corrections >2 m.
2. The safety check rejects the optimizer's correction and keeps the drifting IMU prediction.
3. The next optimization sees an even larger discrepancy (since the IMU continued to drift).
4. All subsequent corrections are also rejected → system is locked into pure dead-reckoning.

In practice, the first rejection happened around frame 80, and from that point on **every single optimization was rejected**. The system diverged to tens of kilometers.

### Fix

Removed the position jump rejection. The optimizer's converged solution should be trusted. The safety check is replaced with a logging statement for monitoring:

**`src/optimization/VioOptimizer.cpp`:**
```cpp
// Log position correction magnitude (no rejection — trust the optimizer)
double jump = (optimized_pos - pre_opt_pos).norm();
if (jump > 1.0) {
    std::printf("[VioOptimizer] correction: %.2fm\n", jump);
}
```

### Impact

With the rejection removed, the optimizer can freely correct the latest pose to be consistent with the sliding window. Corrections of 1-7 m are normal as the gauge frame shifts.

---

## Current State and Remaining Issues

### Results on EuRoC V1_01_easy

| Metric | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| Final position error | ~94,000 m | **4.3 m** |
| Horizontal error (x,y) | catastrophic | **0.31 m** |
| Vertical error (z) | catastrophic | **4.32 m** |
| Reprojection error (2 KF) | ~1.6 px | ~0.11 px |

### Remaining vertical drift

The 4.3 m drift is almost entirely in Z (vertical). This is consistent with accelerometer bias drift — without marginalization, the system has no long-term memory to constrain the vertical direction.

### Marginalization is disabled

The Schur complement marginalization prior was tested but currently causes a secondary failure mode: the prior becomes too stiff over time and fights against vision observations, eventually causing all landmarks to be pruned (>15 px error threshold), which leaves the system blind. The prior then dominates a vision-less optimization, driving divergence.

The system currently uses simple window sliding (drop oldest frame, fix new oldest as gauge). This sacrifices information from dropped frames but avoids the toxic prior problem.

Fixing marginalization is the next priority for achieving sub-meter accuracy.

---

## v0.2.0 Summary of Files Changed

| File | Change |
|------|--------|
| `src/main.mm` | Switched to `cv::fisheye::stereoRectify` / `initUndistortRectifyMap`; added KF0 observation registration |
| `src/metal_main.mm` | Same fisheye and KF0 observation fixes |
| `src/optimization/VioOptimizer.cpp` | Removed position jump rejection; restored landmark pruning (15 px); simple window sliding (marginalization disabled) |
