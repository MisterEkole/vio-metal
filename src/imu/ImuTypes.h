#pragma once

#include <Eigen/Dense>

namespace vio {

struct ImuNoiseParams {
    double gyro_noise_density = 1.6968e-04;     // rad/s/√Hz
    double accel_noise_density = 2.0000e-03;     // m/s²/√Hz
    double gyro_random_walk = 1.9393e-05;        // rad/s²/√Hz
    double accel_random_walk = 3.0000e-03;       // m/s³/√Hz
};

struct PreintegrationResult {
    Eigen::Quaterniond delta_R;
    Eigen::Vector3d delta_v;
    Eigen::Vector3d delta_p;
    Eigen::Matrix<double, 9, 9> covariance;
    Eigen::Matrix<double, 9, 3> J_bg;
    Eigen::Matrix<double, 9, 3> J_ba;
    double dt;
    Eigen::Vector3d linearized_bg;
    Eigen::Vector3d linearized_ba;
};

} 
