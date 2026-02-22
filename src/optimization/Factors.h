#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "core/Types.h"
#include "imu/ImuTypes.h"

namespace vio {

// ================================================================
// Safe small-angle quaternion from axis-angle.
//
// Uses squaredNorm() instead of norm() to avoid the Jet derivative
// singularity: sqrt'(0) = 1/(2·sqrt(0)) = Inf → NaN propagation.
// This matters when bias corrections are zero.
// ================================================================
template <typename T>
Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
    T theta_sq = theta.squaredNorm();  // No sqrt → no singularity

    if (theta_sq < T(1e-12)) {
        // First-order: q ≈ [1, θ/2], then normalize
        return Eigen::Quaternion<T>(
            T(1.0), theta.x() * T(0.5),
            theta.y() * T(0.5), theta.z() * T(0.5)).normalized();
    }

    T theta_norm = sqrt(theta_sq);
    T half_theta = theta_norm * T(0.5);
    Eigen::Matrix<T, 3, 1> axis = theta / theta_norm;
    return Eigen::Quaternion<T>(
        cos(half_theta),
        axis.x() * sin(half_theta),
        axis.y() * sin(half_theta),
        axis.z() * sin(half_theta));
}

// ================================================================
// IMU Preintegration Factor — 15D residual
// Connects two consecutive keyframe states.
//
// This is a functor struct, NOT a CostFunction. It must be wrapped
// with NumericDiffCostFunction (not AutoDiff) because quaternion/
// Lie group operations produce NaN derivatives in Ceres Jet types
// when bias corrections are near zero (sqrt(0) singularity).
// ================================================================
struct ImuFactor {
    ImuFactor(const PreintegrationResult& preint, const Eigen::Vector3d& gravity)
        : preint_(preint), g_(gravity)
    {
        Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();
        cov.block<9, 9>(0, 0) = preint.covariance;

        double dt = std::max(preint.dt, 1e-4);
        double gyro_rw_var  = 1.9393e-05 * 1.9393e-05 * dt;
        double accel_rw_var = 3.0000e-03 * 3.0000e-03 * dt;
        cov.block<3, 3>(9, 9)   = Eigen::Matrix3d::Identity() * gyro_rw_var;
        cov.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * accel_rw_var;

        cov.diagonal().array() += 1e-6;

        auto llt = cov.llt();
        if (llt.info() == Eigen::Success) {
            sqrt_info_ = llt.matrixL().solve(
                Eigen::Matrix<double, 15, 15>::Identity());
        } else {
            sqrt_info_ = Eigen::Matrix<double, 15, 15>::Identity() * 100.0;
        }
    }

    // Called by NumericDiffCostFunction with T = double only
    bool operator()(const double* const pi, const double* const qi,
                    const double* const vba_i,
                    const double* const pj, const double* const qj,
                    const double* const vba_j,
                    double* residuals) const
    {
        Eigen::Map<Eigen::Matrix<double, 15, 1>> res(residuals);
        res.setZero();

        if (preint_.dt < 1e-7) return true;

        Eigen::Map<const Eigen::Vector3d> p_i(pi), p_j(pj);
        Eigen::Map<const Eigen::Quaterniond> q_i(qi), q_j(qj);

        Eigen::Map<const Eigen::Matrix<double, 9, 1>> vi_map(vba_i), vj_map(vba_j);

        Eigen::Vector3d v_i  = vi_map.head<3>();
        Eigen::Vector3d bg_i = vi_map.segment<3>(3);
        Eigen::Vector3d ba_i = vi_map.segment<3>(6);
        Eigen::Vector3d v_j  = vj_map.head<3>();
        Eigen::Vector3d bg_j = vj_map.segment<3>(3);
        Eigen::Vector3d ba_j = vj_map.segment<3>(6);

        double dt = preint_.dt;

        // Bias correction
        Eigen::Vector3d dbg = bg_i - preint_.linearized_bg;
        Eigen::Vector3d dba = ba_i - preint_.linearized_ba;

        // Corrected preintegrated measurements
        Eigen::Vector3d dR_dbg = preint_.J_bg.block<3, 3>(0, 0) * dbg;
        double angle = dR_dbg.norm();
        Eigen::Quaterniond dq_dbg;
        if (angle < 1e-10) {
            dq_dbg = Eigen::Quaterniond(
                1.0, 0.5 * dR_dbg.x(), 0.5 * dR_dbg.y(), 0.5 * dR_dbg.z());
        } else {
            double ha = angle * 0.5;
            Eigen::Vector3d ax = dR_dbg / angle;
            dq_dbg = Eigen::Quaterniond(
                cos(ha), ax.x() * sin(ha), ax.y() * sin(ha), ax.z() * sin(ha));
        }
        dq_dbg.normalize();

        Eigen::Quaterniond dR_corr = (preint_.delta_R * dq_dbg).normalized();
        Eigen::Vector3d dv_corr = preint_.delta_v
            + preint_.J_bg.block<3, 3>(3, 0) * dbg
            + preint_.J_ba.block<3, 3>(3, 0) * dba;
        Eigen::Vector3d dp_corr = preint_.delta_p
            + preint_.J_bg.block<3, 3>(6, 0) * dbg
            + preint_.J_ba.block<3, 3>(6, 0) * dba;

        // Rotation residual (3D)
        Eigen::Quaterniond q_err = dR_corr.inverse() * (q_i.inverse() * q_j);
        double sign = q_err.w() < 0 ? -1.0 : 1.0;
        res.segment<3>(0) = 2.0 * sign * q_err.vec();

        // Velocity residual (3D)
        res.segment<3>(3) = q_i.inverse() * (v_j - v_i - g_ * dt) - dv_corr;

        // Position residual (3D)
        res.segment<3>(6) = q_i.inverse() * (p_j - p_i - v_i * dt
                           - 0.5 * g_ * dt * dt) - dp_corr;

        // Bias random walk residuals (3D + 3D)
        res.segment<3>(9)  = bg_j - bg_i;
        res.segment<3>(12) = ba_j - ba_i;

        // Weight by information
        res = sqrt_info_ * res;
        return true;
    }

    PreintegrationResult preint_;
    Eigen::Vector3d g_;
    Eigen::Matrix<double, 15, 15> sqrt_info_;
};

// ================================================================
// Monocular Reprojection Factor — 2D residual (AutoDiff-safe)
// ================================================================
struct MonoReprojectionFactor {
    MonoReprojectionFactor(const Eigen::Vector2d& obs,
                           double fx, double fy, double cx, double cy,
                           const Eigen::Matrix4d& T_bc,
                           const Eigen::Matrix2d& sqrt_info)
        : obs_(obs), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          T_bc_(T_bc), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p, const T* const q, const T* const l,
                    T* residuals) const
    {
        Eigen::Map<Eigen::Matrix<T, 2, 1>> res(residuals);
        res.setZero();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(p);
        Eigen::Map<const Eigen::Quaternion<T>> q_w(q);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> l_w(l);

        // World → body
        Eigen::Matrix<T, 3, 1> p_b = q_w.inverse() * (l_w - p_w);

        // Body → camera (invert camera→body)
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = R_bc.transpose() * (p_b - t_bc);

        if (p_c.z() < T(0.01)) return true;

        T inv_z = T(1.0) / p_c.z();
        res << T(fx_) * p_c.x() * inv_z + T(cx_) - T(obs_.x()),
               T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_.y());

        res = sqrt_info_.template cast<T>() * res;
        return true;
    }

    Eigen::Vector2d obs_;
    double fx_, fy_, cx_, cy_;
    Eigen::Matrix4d T_bc_;
    Eigen::Matrix2d sqrt_info_;
};

// ================================================================
// Stereo Reprojection Factor — 4D residual (AutoDiff-safe)
// ================================================================
struct StereoReprojectionFactor {
    StereoReprojectionFactor(const Eigen::Vector2d& obs_l,
                              const Eigen::Vector2d& obs_r,
                              double fx, double fy, double cx, double cy,
                              double baseline, const Eigen::Matrix4d& T_bc,
                              const Eigen::Matrix4d& sqrt_info)
        : obs_l_(obs_l), obs_r_(obs_r), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          baseline_(baseline), T_bc_(T_bc), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p, const T* const q, const T* const l,
                    T* residuals) const
    {
        Eigen::Map<Eigen::Matrix<T, 4, 1>> res(residuals);
        res.setZero();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(p);
        Eigen::Map<const Eigen::Quaternion<T>> q_w(q);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> l_w(l);

        // World → body
        Eigen::Matrix<T, 3, 1> p_b = q_w.inverse() * (l_w - p_w);

        // Body → camera (invert camera→body)
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = R_bc.transpose() * (p_b - t_bc);

        if (p_c.z() < T(0.01)) return true;

        T inv_z = T(1.0) / p_c.z();
        res[0] = T(fx_) * p_c.x() * inv_z + T(cx_) - T(obs_l_.x());
        res[1] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_l_.y());
        res[2] = T(fx_) * (p_c.x() - T(baseline_)) * inv_z + T(cx_) - T(obs_r_.x());
        res[3] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_r_.y());

        res = sqrt_info_.template cast<T>() * res;
        return true;
    }

    Eigen::Vector2d obs_l_, obs_r_;
    double fx_, fy_, cx_, cy_, baseline_;
    Eigen::Matrix4d T_bc_, sqrt_info_;
};

} // namespace vio
