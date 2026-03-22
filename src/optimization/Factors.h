#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "core/Types.h"
#include "imu/ImuTypes.h"

namespace vio {

template <typename T>
Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
    T theta_sq = theta.squaredNorm();
    if (theta_sq < T(1e-12)) {
        return Eigen::Quaternion<T>(T(1.0), theta.x()*T(0.5), theta.y()*T(0.5), theta.z()*T(0.5)).normalized();
    }
    T theta_norm = sqrt(theta_sq);
    T half_theta = theta_norm * T(0.5);
    Eigen::Matrix<T, 3, 1> axis = theta / theta_norm;
    return Eigen::Quaternion<T>(cos(half_theta), axis.x()*sin(half_theta), axis.y()*sin(half_theta), axis.z()*sin(half_theta));
}

struct ImuFactor {
    ImuFactor(const PreintegrationResult& preint, const Eigen::Vector3d& gravity)
        : preint_(preint), g_(gravity) {
        Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();
        cov.block<9, 9>(0, 0) = preint.covariance;
        double dt = std::max(preint.dt, 1e-4);
        cov.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (1.9393e-05 * 1.9393e-05 * dt);
        cov.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (3.0000e-03 * 3.0000e-03 * dt);
        cov.diagonal().array() += 1e-6;

        auto llt = cov.llt();
        Eigen::Matrix<double, 15, 15> I15 = Eigen::Matrix<double, 15, 15>::Identity();
        if (llt.info() == Eigen::Success) {
            sqrt_info_ = llt.matrixL().solve(I15);
        } else {
            // Fallback: diagonal inverse
            Eigen::VectorXd diag = cov.diagonal();
            for (int i = 0; i < 15; i++) {
                sqrt_info_(i, i) = 1.0 / std::sqrt(std::max(diag(i), 1e-6));
            }
        }
    }

    template <typename T>
    bool operator()(const T* const pi, const T* const qi, const T* const vba_i,
                    const T* const pj, const T* const qj, const T* const vba_j,
                    T* residuals) const {
        if (preint_.dt < 1e-7) return true;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pi), p_j(pj);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(qi), q_j(qj);
        Eigen::Map<const Eigen::Matrix<T, 9, 1>> vi_map(vba_i), vj_map(vba_j);

        Eigen::Matrix<T, 3, 1> v_i = vi_map.template head<3>();
        Eigen::Matrix<T, 3, 1> bg_i = vi_map.template segment<3>(3);
        Eigen::Matrix<T, 3, 1> ba_i = vi_map.template segment<3>(6);
        Eigen::Matrix<T, 3, 1> v_j = vj_map.template head<3>();
        Eigen::Matrix<T, 3, 1> bg_j = vj_map.template segment<3>(3);
        Eigen::Matrix<T, 3, 1> ba_j = vj_map.template segment<3>(6);

        Eigen::Matrix<T, 3, 1> dbg = bg_i - preint_.linearized_bg.cast<T>();
        Eigen::Matrix<T, 3, 1> dba = ba_i - preint_.linearized_ba.cast<T>();

        Eigen::Matrix<T, 3, 1> theta = preint_.J_bg.block<3, 3>(0, 0).cast<T>() * dbg;
        Eigen::Quaternion<T> dq_dbg = deltaQ(theta);
        Eigen::Quaternion<T> dR_corr = (preint_.delta_R.cast<T>() * dq_dbg).normalized();

        Eigen::Matrix<T, 3, 1> dv_corr = preint_.delta_v.cast<T>() + preint_.J_bg.block<3,3>(3,0).cast<T>()*dbg + preint_.J_ba.block<3,3>(3,0).cast<T>()*dba;
        Eigen::Matrix<T, 3, 1> dp_corr = preint_.delta_p.cast<T>() + preint_.J_bg.block<3,3>(6,0).cast<T>()*dbg + preint_.J_ba.block<3,3>(6,0).cast<T>()*dba;

        Eigen::Map<Eigen::Matrix<T, 15, 1>> res(residuals);
        Eigen::Quaternion<T> q_err = dR_corr.inverse() * (q_i.inverse() * q_j);
        T sign = q_err.w() < T(0) ? T(-1.0) : T(1.0);
        res.template segment<3>(0) = T(2.0) * sign * q_err.vec();
        res.template segment<3>(3) = q_i.inverse() * (v_j - v_i - g_.cast<T>() * T(preint_.dt)) - dv_corr;
        res.template segment<3>(6) = q_i.inverse() * (p_j - p_i - v_i * T(preint_.dt) - T(0.5) * g_.cast<T>() * T(preint_.dt) * T(preint_.dt)) - dp_corr;
        res.template segment<3>(9) = bg_j - bg_i;
        res.template segment<3>(12) = ba_j - ba_i;

        res = sqrt_info_.cast<T>() * res;
        return true;
    }

    PreintegrationResult preint_;
    Eigen::Vector3d g_;
    Eigen::Matrix<double, 15, 15> sqrt_info_;
};

struct MonoReprojectionFactor {
    MonoReprojectionFactor(const Eigen::Vector2d& obs, double fx, double fy, double cx, double cy,
                           const Eigen::Matrix4d& T_bc, const Eigen::Matrix2d& sqrt_info)
        : obs_(obs), fx_(fx), fy_(fy), cx_(cx), cy_(cy), T_bc_(T_bc), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p, const T* const q, const T* const l, T* residuals) const {
        Eigen::Map<Eigen::Matrix<T, 2, 1>> res(residuals);

        Eigen::Matrix<T, 3, 1> p_w = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(l);
        Eigen::Matrix<T, 3, 1> t_wb = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(p);
        Eigen::Quaternion<T> q_wb = Eigen::Map<const Eigen::Quaternion<T>>(q);

        // World → body → camera
        Eigen::Matrix<T, 3, 1> p_b = q_wb.inverse() * (p_w - t_wb);
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = R_bc * p_b + t_bc;

        if (p_c.z() < T(1e-4)) {
            res.setZero();
            return true;
        }

        // Pinhole projection
        T inv_z = T(1.0) / p_c.z();
        res[0] = T(fx_) * p_c.x() * inv_z + T(cx_) - T(obs_.x());
        res[1] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_.y());

        res = sqrt_info_.template cast<T>() * res;
        return true;
    }

    Eigen::Vector2d obs_;
    double fx_, fy_, cx_, cy_;
    Eigen::Matrix4d T_bc_;
    Eigen::Matrix2d sqrt_info_;
};

struct StereoReprojectionFactor {
    StereoReprojectionFactor(const Eigen::Vector2d& obs_l, const Eigen::Vector2d& obs_r,
                              double fx, double fy, double cx, double cy,
                              double baseline, const Eigen::Matrix4d& T_bc,
                              const Eigen::Matrix4d& sqrt_info)
        : obs_l_(obs_l), obs_r_(obs_r), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          baseline_(baseline), T_bc_(T_bc), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p, const T* const q, const T* const l, T* residuals) const {
        Eigen::Map<Eigen::Matrix<T, 4, 1>> res(residuals);

        // World → body → camera
        Eigen::Matrix<T, 3, 1> p_w = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(l);
        Eigen::Matrix<T, 3, 1> t_wb = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(p);
        Eigen::Quaternion<T> q_wb = Eigen::Map<const Eigen::Quaternion<T>>(q);
        Eigen::Matrix<T, 3, 1> p_b = q_wb.inverse() * (p_w - t_wb);
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = R_bc * p_b + t_bc;

        if (p_c.z() < T(1e-4)) {
            res.setZero();
            return true;
        }

        // Stereo pinhole projection
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

struct LoopClosureFactor {
    LoopClosureFactor(const Eigen::Vector3d& rel_p,
                      const Eigen::Quaterniond& rel_q,
                      const Eigen::Matrix<double, 6, 6>& sqrt_info)
        : rel_p_(rel_p), rel_q_(rel_q), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const pi, const T* const qi,
                    const T* const pj, const T* const qj,
                    T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pi), p_j(pj);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(qi), q_j(qj);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);

        // Relative transform error: T_ij = T_i^{-1} * T_j
        Eigen::Matrix<T, 3, 1> dp = q_i.inverse() * (p_j - p_i);
        Eigen::Quaternion<T> dq = q_i.inverse() * q_j;

        res.template segment<3>(0) = dp - rel_p_.template cast<T>();
        Eigen::Quaternion<T> dq_err = rel_q_.template cast<T>().inverse() * dq;
        if (dq_err.w() < T(0)) { dq_err.coeffs() *= T(-1); }
        res.template segment<3>(3) = T(2.0) * dq_err.vec();

        res = sqrt_info_.template cast<T>() * res;
        return true;
    }

    Eigen::Vector3d rel_p_;
    Eigen::Quaterniond rel_q_;
    Eigen::Matrix<double, 6, 6> sqrt_info_;
};

} 