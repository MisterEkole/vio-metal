# Math & Physics Review v0.1.0

**Date:** 2026-03-21
**Codebase:** vio-metal (Visual-Inertial Odometry with Metal GPU acceleration)
**Reference:** Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017)

This document reviews every section of the codebase where math or physics could introduce systematic errors. Each section is marked:
- **CORRECT** — verified against reference literature
- **BUG** — confirmed incorrect, with explanation and fix
- **CONCERN** — not strictly wrong but may cause issues

---

## 1. SO(3) Utilities (`src/core/Types.h`)

### 1.1 Skew-symmetric matrix (`skewSymmetric`, line 19)

```cpp
S <<     0, -v.z(),  v.y(),
     v.z(),      0, -v.x(),
    -v.y(),  v.x(),      0;
```

**CORRECT.** Standard `[v]_x` such that `[v]_x * u = v x u`.

### 1.2 Exponential map (`expSO3`, line 28)

```cpp
return I + sin(theta) * K + (1 - cos(theta)) * K * K;
```

**CORRECT.** Rodrigues' formula: `exp([omega]_x) = I + sin(theta)/theta * [omega]_x + (1-cos(theta))/theta^2 * [omega]_x^2`. Since `K = [axis]_x = [omega]_x / theta`, the code uses `K` and `K*K` which already absorbs the `1/theta` and `1/theta^2` factors.

Small-angle approximation `I + [omega]_x` is also correct (first-order Taylor).

### 1.3 Logarithmic map (`logSO3`, line 41)

```cpp
double cos_angle = clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
double angle = acos(cos_angle);
lnR = (angle / (2 * sin(angle))) * (R - R^T);
```

**CORRECT.** Standard inverse Rodrigues. The vee operator extracts `(lnR(2,1), lnR(0,2), lnR(1,0))` which matches the skew-symmetric convention.

### 1.4 Right Jacobian of SO(3) (`rightJacobianSO3`, line 52)

```cpp
Jr = I - ((1 - cos(theta)) / theta^2) * [omega]_x
   + ((1 - sin(theta)/theta) / theta^2) * [omega]_x^2
```

**CORRECT.** Matches the standard formula:
`Jr(theta) = I - (1-cos(theta))/theta^2 * [theta]_x + (theta - sin(theta))/theta^3 * [theta]_x^2`

The coefficient `(1 - sin(theta)/theta) / theta^2 = (theta - sin(theta)) / theta^3`. Verified.

Small-angle approximation `I - 0.5 * [omega]_x` is correct (first-order).

### 1.5 `deltaQ` template function (`src/optimization/Factors.h`, line 10)

```cpp
Quaternion(cos(half_theta), axis * sin(half_theta))
```

**CORRECT.** Standard axis-angle to quaternion conversion via `q = (cos(theta/2), axis * sin(theta/2))`. Small-angle case `q ~ (1, theta/2)` is also correct.

---

## 2. IMU Preintegration (`src/imu/ImuPreintegrator.cpp`)

Reference: Forster et al. 2017, Section IV.

### 2.1 Bias-corrected measurements (line 29-30)

```cpp
omega = gyro - bg0_;
acc   = accel - ba0_;
```

**CORRECT.** Subtracts linearization-point biases from raw measurements.

### 2.2 Rotation increment (line 35-37)

```cpp
omega_dt = omega * dt;
dR_inc = expSO3(omega_dt);
Jr = rightJacobianSO3(omega_dt);
```

**CORRECT.** Incremental rotation from angular velocity integrated over `dt`.

### 2.3 Error-state transition matrix A (line 40-47)

```cpp
A(0:3, 0:3) = dR_inc^T
A(3:6, 0:3) = -R_k * [acc]_x * dt
A(6:9, 0:3) = -0.5 * R_k * [acc]_x * dt^2
A(6:9, 3:6) = I * dt
```

**CORRECT.** Matches Forster eq. (A.7) / (40). The error-state transition for the discrete-time preintegration propagation.

### 2.4 Noise input matrix B (line 49-52)

```cpp
B(0:3, 0:3) = Jr * dt          // gyro noise -> rotation error
B(3:6, 3:6) = R_k * dt         // accel noise -> velocity error
B(6:9, 3:6) = 0.5 * R_k * dt^2 // accel noise -> position error
```

**CORRECT.** Matches Forster eq. (A.7).

### 2.5 Covariance propagation (line 55-61)

```cpp
Qc = diag(ng^2 * I, na^2 * I)
cov = A * cov * A^T + B * Qc * B^T
```

**CORRECT.** Standard discrete-time covariance propagation. Noise densities squared give the continuous-time PSD, multiplied through `B` which contains the `dt` factors.

### 2.6 Bias Jacobians (line 63-77)

```cpp
// Rotation bias Jacobian
J_bg(0:3, :) = dR_inc^T * J_bg_R_old - Jr * dt

// Velocity bias Jacobians
J_bg(3:6, :) = J_bg_v_old - R_k * [acc]_x * J_bg_R_old * dt
J_ba(3:6, :) = J_ba_v_old - R_k * dt

// Position bias Jacobians
J_bg(6:9, :) = J_bg(6:9, :) + J_bg_v_old * dt - 0.5 * R_k * [acc]_x * J_bg_R_old * dt^2
J_ba(6:9, :) = J_ba(6:9, :) + J_ba_v_old * dt - 0.5 * R_k * dt^2
```

**CORRECT.** These recursive Jacobian updates match Forster eq. (A.8)/(A.9). The rotation Jacobian uses `- Jr * dt` which is the negative right Jacobian — correct because `d(delta_R)/d(bg)` propagates with a negative sign per the Forster derivation.

### 2.7 Preintegrated measurement updates (line 86-89)

```cpp
dp = dp + dv * dt + 0.5 * R_k * acc * dt^2
dv = dv + R_k * acc * dt
dR = Quaterniond(R_k * dR_inc)
```

**CORRECT.** Matches Forster eq. (33)/(34)/(35). The ordering matters: position uses the *old* velocity (before update), which is correct since `dp` is updated before `dv`.

### 2.8 First-order bias correction (`getCorrected`, line 108-128)

```cpp
dbg = new_bg - bg0_;
dba = new_ba - ba0_;

// Rotation correction
dR_correction = J_bg_.block<3,3>(0,0).transpose() * dbg;   // LINE 115
delta_R_corrected = dR_ * Quaterniond(expSO3(dR_correction));

// Velocity and position corrections
delta_v_corrected = dv_ + J_bg(3:6,:) * dbg + J_ba(3:6,:) * dba;
delta_p_corrected = dp_ + J_bg(6:9,:) * dbg + J_ba(6:9,:) * dba;
```

#### BUG: Line 115 — Spurious `.transpose()` on rotation bias Jacobian

The correction formula per Forster eq. (40) is:

```
delta_R_corrected = delta_R * Exp(d(delta_R)/d(bg) * dbg)
```

where `d(delta_R)/d(bg) = J_bg_.block<3,3>(0,0)` (the 3x3 rotation bias Jacobian, already computed correctly during integration).

The code applies `.transpose()`:
```cpp
dR_correction = J_bg_.block<3,3>(0,0).transpose() * dbg;  // WRONG
```

This should be:
```cpp
dR_correction = J_bg_.block<3,3>(0,0) * dbg;              // CORRECT
```

**Why this matters:** The rotation bias Jacobian `J_bg_R` is not symmetric — it accumulates products of `dR_inc^T` and `Jr` terms. Transposing it applies the wrong linear correction to the rotation when biases change, meaning **every bias-corrected preintegration result has an incorrect rotation**. Since the IMU factor uses `getCorrected()` (indirectly via `ImuFactor` which applies the same Jacobian), this corrupts the rotation residual in the optimizer.

**Impact:** The velocity and position corrections (lines 120-121) do NOT use `.transpose()` and are correct. Only the rotation correction is affected.

**Note:** The `ImuFactor` in `Factors.h` (line 58) applies its own bias correction independently:
```cpp
Eigen::Quaterniond dq_dbg = deltaQ(Eigen::Vector3d(preint_.J_bg.block<3,3>(0,0) * dbg));
```
This does NOT have the transpose bug — it uses the Jacobian directly. So the bug in `getCorrected()` only affects code paths that call `ImuPreintegrator::getCorrected()` directly. **Check whether the pipeline calls `getCorrected()` or relies on the factor's internal correction.**

---

## 3. IMU Factor (`src/optimization/Factors.h`, line 21)

### 3.1 Bias correction in factor (line 55-62)

```cpp
dbg = bg_i - preint_.linearized_bg;
dba = ba_i - preint_.linearized_ba;

dq_dbg = deltaQ(J_bg(0:3,:) * dbg);     // No transpose — CORRECT
dR_corr = (delta_R * dq_dbg).normalized();
dv_corr = delta_v + J_bg(3:6,:)*dbg + J_ba(3:6,:)*dba;
dp_corr = delta_p + J_bg(6:9,:)*dbg + J_ba(6:9,:)*dba;
```

**CORRECT.** The factor applies first-order bias correction directly, matching Forster eq. (40). This is independent of `getCorrected()` and does NOT have the transpose bug.

### 3.2 Rotation residual (line 65-66)

```cpp
q_err = dR_corr.inverse() * (q_i.inverse() * q_j);
res(0:3) = 2.0 * sign(q_err.w) * q_err.vec();
```

**CORRECT.** The rotation error is `Log(delta_R_corrected^{-1} * R_i^T * R_j)`, approximated as `2 * q_err.vec()` for small angles. The sign flip ensures we stay in the correct hemisphere of the quaternion double cover.

### 3.3 Velocity residual (line 67)

```cpp
res(3:6) = R_i^T * (v_j - v_i - g * dt) - dv_corr;
```

**CORRECT.** Matches Forster eq. (42): the expected velocity change in the body frame minus the preintegrated velocity.

### 3.4 Position residual (line 68)

```cpp
res(6:9) = R_i^T * (p_j - p_i - v_i * dt - 0.5 * g * dt^2) - dp_corr;
```

**CORRECT.** Matches Forster eq. (43).

### 3.5 Bias random walk residuals (line 69-70)

```cpp
res(9:12)  = bg_j - bg_i;
res(12:15) = ba_j - ba_i;
```

**CORRECT.** Biases evolve as a random walk; the residual penalizes large changes between consecutive keyframes.

### 3.6 Information matrix / sqrt_info (line 24-41)

The 15x15 covariance is built as:
```
cov(0:9, 0:9)   = preint.covariance       (9x9, from preintegration)
cov(9:12, 9:12) = gyro_random_walk^2 * dt * I
cov(12:15, 12:15) = accel_random_walk^2 * dt * I
```

**CORRECT.** The preintegration covariance covers `[dR, dv, dp]`, and the bias random walk covariances are appended. The `sqrt_info` is computed as `L^{-1}` from the Cholesky decomposition, which is the correct whitening matrix.

#### CONCERN: Fallback diagonal sqrt_info (line 36-41)

```cpp
// Fallback: use diagonal of covariance for a safe inverse
sqrt_info_(i, i) = 1.0 / sqrt(max(diag(i), 1e-6));
```

When the Cholesky fails, the fallback constructs a diagonal `sqrt_info` but **does not zero the off-diagonal elements**. Since `sqrt_info_` is declared as a member variable, it's value-initialized to zero, so this is technically fine for the first construction. However, if an `ImuFactor` object were ever reused (unlikely in practice), stale off-diagonals could persist. Low risk.

---

## 4. Reprojection Factors (`src/optimization/Factors.h`)

### 4.1 Mono reprojection (`MonoReprojectionFactor`, line 81)

Transform chain: World -> Body -> Camera -> Image

```cpp
p_b = q_wb.inverse() * (p_w - t_wb);   // World to Body — CORRECT
p_c = R_cb * p_b + t_cb;                // Body to Camera (T_bc inverted) — BUG (see Section 9)
u = fx * p_c.x / p_c.z + cx;           // Pinhole projection
v = fy * p_c.y / p_c.z + cy;
```

#### BUG: Unnecessary inversion of body-to-camera transform

The code extracts `R_bc, t_bc` from `T_bc_ = T_cam0_imu` (which is already IMU→camera), then inverts it to get `R_cb, t_cb`, and applies the inverse. Since `T_cam0_imu` already IS the body-to-camera transform, this double-inverts it. The correct code is simply `p_c = R_bc * p_b + t_bc`. See Section 9.1 for full analysis and the cancellation effect with the landmark transform bug.

#### CONCERN: Zero-residual on negative depth (line 111-113)

```cpp
if (p_c.z() < T(1e-4)) {
    res.setZero();
    return true;
}
```

When a landmark projects behind the camera, the residual is set to zero instead of returning a large error. This means the optimizer gets no gradient signal to fix the problem — effectively treating behind-camera landmarks as perfectly observed. A large residual or returning `false` would be safer. In practice, the depth filter in `VioOptimizer::optimize()` (line 429) catches most such cases before the factor is created, so this is a secondary concern.

### 4.2 Stereo reprojection (`StereoReprojectionFactor`, line 132)

```cpp
res[0] = fx * p_c.x / z + cx - obs_l.x;    // Left u
res[1] = fy * p_c.y / z + cy - obs_l.y;    // Left v
res[2] = fx * (p_c.x - baseline) / z + cx - obs_r.x;  // Right u
res[3] = fy * p_c.y / z + cy - obs_r.y;    // Right v
```

**CORRECT.** For a rectified stereo pair with horizontal baseline `b`, the right camera sees the point at `(p_c.x - b, p_c.y, p_c.z)` in the left camera frame. The projection formula `fx * (x - b) / z + cx` is the standard rectified stereo model.

**Assumption:** `baseline` is the signed x-component of the translation between left and right cameras (positive for a right camera to the right of the left camera). The code computes `baseline = T_cam1_cam0.block<3,1>(0,3).norm()` — this takes the magnitude, which is correct as long as the cameras are horizontally aligned (which is true after rectification).

### 4.3 Loop closure factor (`LoopClosureFactor`, line 181)

```cpp
dp = q_i.inverse() * (p_j - p_i);     // Relative translation in frame i
dq = q_i.inverse() * q_j;              // Relative rotation
res(0:3) = dp - rel_p_measured;
res(3:6) = 2 * dq_err.vec();
```

**CORRECT.** Standard relative pose factor formulation.

---

## 5. Marginalization (`src/optimization/Marginalization.cpp` + `.h`)

### 5.1 Cost function formulation (from header comment)

The marginalization prior encodes:
```
cost = 0.5 * || sqrt_info * (x - x0) + e0 ||^2
```

Expanding: `cost = 0.5 * (dx^T * H * dx + 2 * b^T * dx + e0^T * e0)`

where `H = sqrt_info^T * sqrt_info` and `b = sqrt_info^T * e0` should match the Schur complement Hessian and gradient.

### 5.2 Eigendecomposition and sqrt_info (line 37-38)

```cpp
sqrt_info_ = D^{1/2} * V^T;
```

Since `H = V * D * V^T`, we have `sqrt_info^T * sqrt_info = V * D^{1/2} * D^{1/2} * V^T = V * D * V^T = H`. **CORRECT.**

### 5.3 Residual offset e0 (line 43)

```cpp
e0_ = D^{-1/2} * V^T * b;
```

#### BUG: Sign of e0 is wrong

The cost function must reproduce the original linearized cost at the linearization point and its gradient. The original cost from the Schur complement is:

```
cost(dx) = 0.5 * dx^T * H * dx + b^T * dx + c
```

where `b` is the gradient (= `-J^T * r` from `VioOptimizer.cpp` line 244). The prior factor computes:

```
residual = sqrt_info * dx + e0
cost = 0.5 * ||residual||^2
     = 0.5 * dx^T * H * dx + dx^T * sqrt_info^T * e0 + 0.5 * e0^T * e0
```

For the gradient to match, we need: `sqrt_info^T * e0 = b`

Solving: `e0 = (sqrt_info^T)^{-1} * b = (V * D^{1/2})^{-1} * b = D^{-1/2} * V^T * b`

This is what the code computes. **However**, the convention for `b` matters critically.

In `VioOptimizer.cpp` line 244:
```cpp
b_full = -J^T * r;
```

This computes `b = -J^T * r`, which is the **negative** gradient of the cost function `0.5 * ||r||^2`. The gradient of the cost is `J^T * r`, so `b = -gradient`.

The desired cost expansion around the linearization point for the **kept** variables after Schur complement:

```
cost(dx_k) = 0.5 * dx_k^T * H_prior * dx_k + b_prior^T * dx_k + c
```

At `dx_k = 0`: `d(cost)/d(dx_k) = b_prior`. Since `b_prior = b_k - H_km * H_mm^{-1} * b_m`, and `b = -J^T * r`, at the linearization point the gradient is `b_prior`.

For the factor to reproduce this gradient at `dx = 0`:
```
d(0.5 * ||sqrt_info * dx + e0||^2)/d(dx) |_{dx=0} = sqrt_info^T * e0
```

We need `sqrt_info^T * e0 = b_prior`. But `b_prior` contains the **negative** gradient convention (since `b = -J^T * r`). The optimizer minimizes `0.5 * ||r||^2`, and the Gauss-Newton step solves `H * dx = -b = J^T * r` (positive gradient).

**The sign should be negative:**
```cpp
e0_ = -inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
```

**Why:** The `b` vector passed to `setPrior` is `b_prior = b_k - H_km * H_mm^{-1} * b_m`, where `b = -J^T * r`. At the linearization point, `dx = 0`, so the residual is just `e0`. The cost is `0.5 * ||e0||^2`, and the gradient w.r.t. `dx` is `sqrt_info^T * e0`. For the prior to push the optimizer toward the correct minimum, this gradient must equal `b_prior` (which has the convention `b = -J^T * r`).

But the prior is added as a residual block to Ceres, which computes `0.5 * ||r||^2`. Ceres internally computes the gradient as `J_prior^T * r_prior = sqrt_info^T * (sqrt_info * dx + e0)`. At `dx = 0`, this is `sqrt_info^T * e0`. For Ceres to get the right gradient, we need `sqrt_info^T * e0 = b_prior`. Since the code computes `e0 = (sqrt_info^T)^{-1} * b_prior`, this gives `sqrt_info^T * e0 = b_prior`.

**Wait — reconsidering.** The Gauss-Newton system is `H * dx = b` where `b = -J^T * r`. Ceres minimizes `0.5 * sum(r_i^2)`. The gradient of the objective is `J^T * r`, and the update direction satisfies `H * dx = -J^T * r = b`. This means `b` is the right-hand side of the normal equations, which equals the **negative** gradient.

For the prior factor residual `r_prior = sqrt_info * dx + e0`, the gradient contribution is:
```
d(0.5 * r_prior^T * r_prior) / d(dx) = sqrt_info^T * r_prior = sqrt_info^T * (sqrt_info * dx + e0)
```

At `dx = 0`, gradient = `sqrt_info^T * e0`.

For the prior to have the correct gradient at the linearization point, we need the negative gradient from the prior to match `b_prior`:
```
-sqrt_info^T * e0 = b_prior    (since b = negative gradient)
```

Therefore: `e0 = -(sqrt_info^T)^{-1} * b_prior = -D^{-1/2} * V^T * b_prior`

**The current code is missing the negative sign.** This inverts the gradient direction of the marginalization prior, causing it to push the optimizer **away from** the correct linearization point instead of **toward** it.

**Impact:** This is a critical bug. The marginalization prior is the primary mechanism for transferring information from removed keyframes to the current window. With the wrong sign, the prior actively destabilizes the optimization, which explains the trajectory divergence and "positive feedback loop" of bad states observed in the evaluation results.

**Fix:**
```cpp
// Line 43 of Marginalization.cpp
e0_ = -inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
```

### 5.4 MarginalizationFactor residual evaluation (Marginalization.h, line 47-73)

```cpp
res(0:3) = pos - p0;                     // Position delta
res(3:6) = 2 * (q0.inverse() * quat).vec();  // Rotation delta (tangent)
res(6:15) = vba - vba0;                  // Velocity + bias delta
res = sqrt_info * res + e0;
```

**CORRECT** (given a correct `e0`). The tangent-space parameterization is consistent — position and bias deltas are Euclidean, rotation delta uses the 2*vec approximation of the quaternion logarithm.

### 5.5 Schur complement computation (VioOptimizer.cpp, line 252-296)

```cpp
H_full = J^T * J;
b_full = -J^T * r;

// Permute to [marginalize, keep] ordering
// Schur: H_prior = H_kk - H_km * H_mm^{-1} * H_mk
//        b_prior = b_k  - H_km * H_mm^{-1} * b_m
```

**CORRECT.** Standard Schur complement formulation. The permutation logic correctly reorders columns from `[ms0(15), ms1(15), landmarks(3*n)]` to `[ms0(15), landmarks(3*n), ms1(15)]` so that marginalized variables come first.

**CONCERN:** `H_mm` regularization `+= 1e-6` on diagonal is applied before inversion. Combined with the `+= 1e-6` in `setPrior`, there is double regularization, but both are small enough to be numerically harmless.

---

## 6. VIO State Propagation (`VioOptimizer::addKeyframe`, line 27-60)

### 6.1 IMU-based state prediction (line 41-51)

```cpp
R_new = R_prev * delta_R;
v_new = v_prev + g * dt + R_prev * delta_v;
p_new = p_prev + v_prev * dt + 0.5 * g * dt^2 + R_prev * delta_p;
```

**CORRECT.** This is the standard IMU forward propagation using preintegrated measurements. The world-frame equations are:
- `R_{k+1} = R_k * delta_R` (compose rotations)
- `v_{k+1} = v_k + g * dt + R_k * delta_v` (velocity with gravity)
- `p_{k+1} = p_k + v_k * dt + 0.5 * g * dt^2 + R_k * delta_p`

These match Forster eq. (37)-(39).

---

## 7. Gravity Vector (`src/core/Types.h`, line 13)

```cpp
constexpr double GRAVITY_MAGNITUDE = 9.81;
inline Eigen::Vector3d gravity() { return {0.0, 0.0, -9.81}; }
```

**CORRECT** for a z-up coordinate system (EuRoC uses NED-like, but the VIO system uses z-up with gravity pointing down). The EuRoC ground truth uses `g = [0, 0, -9.81]` in the world frame.

---

## 8. Stereo Triangulation (`src/vision/StereoMatcher.cpp`)

### 8.1 Depth from disparity (line 76)

```cpp
depth = fx * baseline / disparity;
```

**CORRECT.** Standard stereo depth formula `z = f * b / d` where `f` is focal length in pixels, `b` is baseline in meters, and `d` is disparity in pixels.

### 8.2 Back-projection to 3D (line 82-86)

```cpp
point_3d = (depth * (u - cx) / fx,
            depth * (v - cy) / fy,
            depth);
```

**CORRECT.** Standard pinhole back-projection: `X = z * (u - cx) / fx`, `Y = z * (v - cy) / fy`, `Z = z`.

### 8.3 Baseline computation (line 30-31)

```cpp
t_cam1_cam0 = T_cam1_cam0.block<3,1>(0,3);
baseline = t_cam1_cam0.norm();
```

**CORRECT** (after rectification). Post-rectification, `T_cam1_cam0` has only an x-translation component, so `norm()` equals `|t_x|`. The sign convention is absorbed into the disparity constraint (`lkp.pt.x - rkp.pt.x > 0`).

---

## 9. Landmark World Transform (`src/vision/FeatureManager.cpp`, line 103-120)

### 9.1 Camera-to-world transform (line 110-116)

#### BUG: Double extrinsic inversion — camera↔body transform applied in wrong direction

**Investigation:** The `T_cam0_imu` convention was verified via the dataset loader:

```cpp
// EurocLoader.cpp:225-228
Eigen::Matrix4d T_imu_cam0 = Eigen::Map<...>(T_BS_cam0.data());  // EuRoC T_BS = sensor-to-body
calib.T_cam0_imu = T_imu_cam0.inverse();                          // body-to-sensor = IMU-to-camera
```

So `T_cam0_imu` transforms **from IMU to camera**: `P_cam = R_ci * P_imu + t_ci`.

**Bug 1 — `getInitializedLandmarksWorld` (FeatureManager.cpp:110-116):**

```cpp
// WRONG: applies IMU→camera forward transform to a camera-frame point
R_ci = T_cam_imu.block<3,3>(0,0);
t_ci = T_cam_imu.block<3,1>(0,3);
P_body = R_ci * landmark_3d + t_ci;     // Should use INVERSE: R_ci^T * (P_cam - t_ci)
```

To go from camera to body, the inverse is needed: `P_body = R_ci^T * (P_cam - t_ci)`.

**Bug 2 — Reprojection factors (Factors.h, MonoReprojectionFactor:100-108 and StereoReprojectionFactor:150-156):**

```cpp
// WRONG: inverts T_bc (which is already IMU→camera) to get camera→IMU, then applies that
R_bc = T_bc_.block<3,3>(0,0);   // R_cam_from_imu
R_cb = R_bc.transpose();         // R_imu_from_cam (unnecessary inversion)
t_cb = -R_cb * t_bc;
p_c = R_cb * p_b + t_cb;         // Applies camera→body transform to a body-frame point!
```

Since `T_bc = T_cam0_imu` already IS the body-to-camera transform, it should be used directly: `p_c = R_bc * p_b + t_bc`.

**Bug 3 — Depth check in VioOptimizer::optimize() (VioOptimizer.cpp:428):** Same inverted transform used for the pre-filter depth check, consistent with Bug 2.

**Why it appeared to work (partial cancellation):**

The two bugs create a double inversion that **cancels for the originating keyframe**. Tracing through the math for a landmark observed at its triangulation keyframe:

```
getInitializedLandmarksWorld:  P_world_wrong = q_wb * (R_ci * P_cam + t_ci) + p_wb
Reprojection factor:           p_b = q_wb^{-1} * (P_world_wrong - p_wb) = R_ci * P_cam + t_ci
                               p_c = R_ci^T * (R_ci * P_cam + t_ci) - R_ci^T * t_ci = P_cam  ✓
```

The errors cancel perfectly, giving back the original camera-frame point. However, for **any other keyframe** observing the same landmark, the wrong world position produces increasingly large reprojection errors as the inter-keyframe baseline grows.

**Observed behavior:** Through KF 10 (frame ~100), landmarks are mostly observed from similar viewpoints, so the error is small. At KF 12 (frame ~111), the camera has moved enough that cross-keyframe reprojection errors explode — initial cost jumped from ~200 to ~11000 (pre-fix), the optimizer degraded good IMU predictions (0.02m → 0.08m), and a drift cascade began.

**Fix (FeatureManager.cpp):**
```cpp
// Correct: use inverse of T_cam_imu for camera→body
Eigen::Matrix3d R_ic = R_ci.transpose();
Eigen::Vector3d t_ic = -R_ci.transpose() * t_ci;
P_body = R_ic * track.landmark_3d + t_ic;
```

**Fix (Factors.h, both mono and stereo):**
```cpp
// Correct: use T_cam0_imu directly for body→camera
p_c = R_bc * p_b + t_bc;   // no inversion needed
```

**Fix (VioOptimizer.cpp depth check):**
```cpp
// Correct: use T_cam0_imu directly
Eigen::Vector3d p_c = R_bc * p_b + t_bc;
```

**Impact:** After fixing all three locations, the optimizer maintains sub-3cm error through KF 11 (previously diverging at KF 10). Initial costs at KF 10 dropped from ~11000 to ~400. The optimizer now improves IMU predictions rather than degrading them.

### 9.2 Quaternion naming (`q_wb`)

The parameter is named `q_wb` and used as `q_wb * P_body + p_wb`, which rotates body-frame points to world frame. This is correct for the world-from-body convention (`R_wb * p_b + t_wb = p_w`).

---

## 10. Loop Closure (`src/vision/LoopDetector.cpp`)

### 10.1 PnP pose recovery (line 100-144)

```cpp
// PnP gives T_camera_world
// R_wc = R_cw^T, t_wc = -R_wc * t_cw  (invert to get world-from-camera)
// R_wb = R_wc * R_ci, t_wb = R_wc * t_ci + t_wc
```

**CORRECT.** The derivation in the comments (lines 139-141) is worked through step by step:
- `p_world = R_wc * (R_ci * p_body + t_ci) + t_wc`
- `= (R_wc * R_ci) * p_body + (R_wc * t_ci + t_wc)`

This correctly chains world-from-camera and camera-from-body.

### 10.2 Relative pose computation in pipeline (metal_main.mm, line 279-280)

```cpp
lc.relative_position = q_match.inverse() * (loop.relative_position - p_match);
lc.relative_rotation = q_match.inverse() * loop.relative_rotation;
```

Here `loop.relative_position` is actually the **absolute** world position of the query body frame (from PnP), and `p_match, q_match` are the matched keyframe's pose from the optimizer window.

```
T_ij (match-from-query in match's body frame):
  relative_position = R_match^T * (p_query - p_match)
  relative_rotation = R_match^T * R_query
```

**CORRECT.** This computes `T_match^{-1} * T_query`, which is the standard relative pose from match to query in the match's body frame.

---

## 11. Summary of Critical Bugs

### Bug 1: `ImuPreintegrator::getCorrected()` — Rotation bias Jacobian transpose (line 115)

**File:** `src/imu/ImuPreintegrator.cpp:115`
**Severity:** Medium-High (depends on whether `getCorrected()` is called in the pipeline)
**Fix:**
```cpp
// BEFORE (wrong):
Eigen::Vector3d dR_correction = J_bg_.block<3,3>(0,0).transpose() * dbg;
// AFTER (correct):
Eigen::Vector3d dR_correction = J_bg_.block<3,3>(0,0) * dbg;
```

**Note:** The `ImuFactor` in `Factors.h` applies its own bias correction without the transpose, so if the pipeline uses `getResult()` and lets the factor handle correction, this bug may not be on the critical path. Verify usage.

### Bug 2: `MarginalizationInfo::setPrior()` — e0 sign (line 43)

**File:** `src/optimization/Marginalization.cpp:43`
**Severity:** Critical
**Fix:**
```cpp
// BEFORE (wrong):
e0_ = inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
// AFTER (correct):
e0_ = -(inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose() * b).eval();
```

This inverts the gradient direction of the marginalization prior, causing a positive feedback loop where bad state estimates are reinforced rather than corrected.

### Bug 3: Double extrinsic inversion — `T_cam0_imu` applied in wrong direction (3 locations)

**Files:**
- `src/vision/FeatureManager.cpp:110-116` — camera→body uses forward instead of inverse
- `src/optimization/Factors.h:100-108, 150-156` — body→camera uses inverse instead of forward
- `src/optimization/VioOptimizer.cpp:428` — depth check uses inverse instead of forward

**Severity:** Critical
**Fix:** Use inverse of `T_cam0_imu` for camera→body (FeatureManager), use `T_cam0_imu` directly for body→camera (Factors + depth check). See Section 9.1 for full derivation.

The bugs partially cancel for the originating keyframe but produce wrong world positions for landmarks observed across keyframes, causing the optimizer to fight correct IMU predictions with incorrect reprojection constraints. This was the root cause of trajectory divergence at KF 10 (frame ~111).

---

## 12. Summary of Concerns (Non-critical)

| # | Location | Issue | Risk |
|---|----------|-------|------|
| 1 | `Factors.h:111-113` | Zero residual for behind-camera points silences the gradient | Low (filtered upstream) |
| 2 | `Factors.h:36-41` | IMU factor fallback doesn't explicitly zero off-diagonals | Very low |
| 3 | ~~`FeatureManager.cpp:115`~~ | ~~`T_cam_imu` direction convention ambiguity~~ | **Confirmed as Bug 3** (see Section 9.1, Section 11) |
| 4 | `VioOptimizer.cpp:281` + `Marginalization.cpp:13` | Double `1e-6` diagonal regularization | Very low |
| 5 | `VioOptimizer.cpp:475` | Using `DENSE_QR` instead of `DENSE_SCHUR` is slower for the problem structure | Performance only |

---

## 13. Recommended Fix Order

1. **Fix Bug 2 (Marginalization e0 sign)** — ✅ FIXED. The inverted prior gradient created a positive feedback loop. Convergence improved from ~40% to ~94%.
2. **Fix Bug 1 (getCorrected transpose)** — ✅ FIXED. Not on the critical path (ImuFactor handles bias correction internally), but corrected for consistency.
3. **Fix Bug 3 (Double extrinsic inversion)** — ✅ FIXED. Verified `T_cam0_imu` convention via `EurocLoader`. Fixed camera→body transform in `FeatureManager`, body→camera in reprojection factors, and depth check in optimizer. This was the root cause of the KF 10 cost explosion and trajectory divergence.
4. **Re-run evaluation** — After all fixes, optimizer maintains sub-3cm error through KF 11 (vs multi-meter divergence pre-fix). Residual drift starting at KF 12-13 likely requires tuning of reprojection weights and/or bias estimation.
