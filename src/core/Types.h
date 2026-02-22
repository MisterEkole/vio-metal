#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <vector>
#include <cmath>

namespace vio {

// Gravity vector (world frame, z-up)
constexpr double GRAVITY_MAGNITUDE = 9.81;
inline Eigen::Vector3d gravity() { return Eigen::Vector3d(0.0, 0.0, -GRAVITY_MAGNITUDE); }

// ============================================================
// SE3 / SO3 Utilities
// ============================================================

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return S;
}

// SO(3) exponential map: axis-angle vector -> rotation matrix
inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity() + skewSymmetric(omega);
    }
    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d K = skewSymmetric(axis);
    return Eigen::Matrix3d::Identity()
         + std::sin(theta) * K
         + (1.0 - std::cos(theta)) * K * K;
}

// SO(3) logarithmic map: rotation matrix -> axis-angle vector
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d& R) {
    double cos_angle = std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
    double angle = std::acos(cos_angle);
    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Matrix3d lnR = (angle / (2.0 * std::sin(angle))) * (R - R.transpose());
    return Eigen::Vector3d(lnR(2,1), lnR(0,2), lnR(1,0));
}

// Right Jacobian of SO(3)
inline Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity() - 0.5 * skewSymmetric(omega);
    }
    Eigen::Vector3d axis = omega / theta;
    Eigen::Matrix3d K = skewSymmetric(axis);
    return Eigen::Matrix3d::Identity()
         - ((1.0 - std::cos(theta)) / (theta * theta)) * skewSymmetric(omega)
         + (1.0 - std::sin(theta) / theta) / (theta * theta) * skewSymmetric(omega) * skewSymmetric(omega);
}

inline Eigen::Vector3d quaternionToRotVec(const Eigen::Quaterniond& q) {
    return logSO3(q.toRotationMatrix());
}

// ============================================================
// Feature observation for optimizer
// ============================================================

struct FeatureObservation {
    uint64_t feature_id;
    Eigen::Vector2d pixel_left;
    Eigen::Vector2d pixel_right;   // (-1,-1) if no stereo
    bool has_stereo = false;
    Eigen::Vector3d landmark_3d;   // World frame initial estimate
    bool landmark_initialized = false;
};

// ============================================================
// Conversion utilities
// ============================================================

inline double nsToSec(uint64_t ns) {
    return static_cast<double>(ns) * 1e-9;
}

inline uint64_t secToNs(double sec) {
    return static_cast<uint64_t>(sec * 1e9);
}

}
