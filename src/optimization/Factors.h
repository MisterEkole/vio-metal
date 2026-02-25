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
            sqrt_info_ = I15 * 100.0;
        }
    }

    bool operator()(const double* const pi, const double* const qi, const double* const vba_i,
                    const double* const pj, const double* const qj, const double* const vba_j,
                    double* residuals) const {
        if (preint_.dt < 1e-7) return true;
        Eigen::Map<const Eigen::Vector3d> p_i(pi), p_j(pj);
        Eigen::Map<const Eigen::Quaterniond> q_i(qi), q_j(qj);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> vi_map(vba_i), vj_map(vba_j);
        
        Eigen::Vector3d v_i = vi_map.head<3>(), bg_i = vi_map.segment<3>(3), ba_i = vi_map.segment<3>(6);
        Eigen::Vector3d v_j = vj_map.head<3>(), bg_j = vj_map.segment<3>(3), ba_j = vj_map.segment<3>(6);

        Eigen::Vector3d dbg = bg_i - preint_.linearized_bg;
        Eigen::Vector3d dba = ba_i - preint_.linearized_ba;

        Eigen::Quaterniond dq_dbg = deltaQ(Eigen::Vector3d(preint_.J_bg.block<3, 3>(0, 0) * dbg));
        Eigen::Quaterniond dR_corr = (preint_.delta_R * dq_dbg).normalized();
        
        Eigen::Vector3d dv_corr = preint_.delta_v + preint_.J_bg.block<3,3>(3,0)*dbg + preint_.J_ba.block<3,3>(3,0)*dba;
        Eigen::Vector3d dp_corr = preint_.delta_p + preint_.J_bg.block<3,3>(6,0)*dbg + preint_.J_ba.block<3,3>(6,0)*dba;

        Eigen::Map<Eigen::Matrix<double, 15, 1>> res(residuals);
        Eigen::Quaterniond q_err = dR_corr.inverse() * (q_i.inverse() * q_j);
        res.segment<3>(0) = 2.0 * (q_err.w() < 0 ? -1.0 : 1.0) * q_err.vec();
        res.segment<3>(3) = q_i.inverse() * (v_j - v_i - g_ * preint_.dt) - dv_corr;
        res.segment<3>(6) = q_i.inverse() * (p_j - p_i - v_i * preint_.dt - 0.5 * g_ * preint_.dt * preint_.dt) - dp_corr;
        res.segment<3>(9) = bg_j - bg_i;
        res.segment<3>(12) = ba_j - ba_i;

        res = sqrt_info_ * res;
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

        //  World-to-Body transform
        Eigen::Matrix<T, 3, 1> p_w = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(l);
        Eigen::Matrix<T, 3, 1> t_wb = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(p);
        Eigen::Quaternion<T> q_wb = Eigen::Map<const Eigen::Quaternion<T>>(q);
        
        // Point in Body Frame: P_b = R_wb^T * (P_w - t_wb)
        Eigen::Matrix<T, 3, 1> p_b = q_wb.inverse() * (p_w - t_wb);
    
        // Body-to-Camera (using your T_bc_ matrix)
        // Extract rotation and translation from the extrinsic matrix
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
    
        // Compute Camera-from-Body: T_cb = T_bc.inverse()
        Eigen::Matrix<T, 3, 3> R_cb = R_bc.transpose();
        Eigen::Matrix<T, 3, 1> t_cb = -R_cb * t_bc;
    
        // Point in Camera Frame
        Eigen::Matrix<T, 3, 1> p_c = R_cb * p_b + t_cb;

        // Check for valid depth (Visibility check)
        if (p_c.z() < T(1e-4)) {
            res.setZero();
            return true;
        }

        // Project to Image Plane (Pinhole Model)
        T inv_z = T(1.0) / p_c.z();
        res[0] = T(fx_) * p_c.x() * inv_z + T(cx_) - T(obs_.x());
        res[1] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_.y());
        
        // Weight by Information Matrix
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

        // World -> Body
        Eigen::Matrix<T, 3, 1> p_w = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(l);
        Eigen::Matrix<T, 3, 1> t_wb = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(p);
        Eigen::Quaternion<T> q_wb = Eigen::Map<const Eigen::Quaternion<T>>(q);
        Eigen::Matrix<T, 3, 1> p_b = q_wb.inverse() * (p_w - t_wb);
    
        // Body -> Camera (Inverting T_bc to get T_cb)
        Eigen::Matrix<T, 3, 3> R_bc = T_bc_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_bc = T_bc_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 3> R_cb = R_bc.transpose();
        Eigen::Matrix<T, 3, 1> t_cb = -R_cb * t_bc;
    
        Eigen::Matrix<T, 3, 1> p_c = R_cb * p_b + t_cb;

        // Check for valid depth (Points must be in front of camera)
        if (p_c.z() < T(1e-4)) {
            res.setZero();
            return true;
        }

        //Project to Stereo Image Plane
        T inv_z = T(1.0) / p_c.z();
        res[0] = T(fx_) * p_c.x() * inv_z + T(cx_) - T(obs_l_.x());
        res[1] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_l_.y());
        res[2] = T(fx_) * (p_c.x() - T(baseline_)) * inv_z + T(cx_) - T(obs_r_.x());
        res[3] = T(fy_) * p_c.y() * inv_z + T(cy_) - T(obs_r_.y());

        // Weight by uncertainty
        res = sqrt_info_.template cast<T>() * res;
        return true;
    }

    Eigen::Vector2d obs_l_, obs_r_;
    double fx_, fy_, cx_, cy_, baseline_;
    Eigen::Matrix4d T_bc_, sqrt_info_;
};

} 