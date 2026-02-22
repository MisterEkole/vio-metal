#pragma once

#include <Eigen/Dense>

namespace vio {

struct ImuNoiseParams {
    double gyro_noise_density = 1.6968e-04;     // rad/s/√Hz
    double accel_noise_density = 2.0000e-03;     // m/s²/√Hz
    double gyro_random_walk = 1.9393e-05;        // rad/s²/√Hz
    double accel_random_walk = 3.0000e-03;       // m/s³/√Hz
};

// Preintegration result for use in optimizer
struct PreintegrationResult {
    Eigen::Quaterniond delta_R;      // Relative rotation
    Eigen::Vector3d delta_v;         // Relative velocity
    Eigen::Vector3d delta_p;         // Relative position
    Eigen::Matrix<double, 9, 9> covariance;  // [dR, dv, dp]
    Eigen::Matrix<double, 9, 3> J_bg;        // Jacobian w.r.t. gyro bias
    Eigen::Matrix<double, 9, 3> J_ba;        // Jacobian w.r.t. accel bias
    double dt;                       // Total integration time
    Eigen::Vector3d linearized_bg;   // Bias at linearization
    Eigen::Vector3d linearized_ba;
};

} 
