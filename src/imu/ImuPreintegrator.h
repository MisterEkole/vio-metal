#pragma once

#include <Eigen/Dense>
#include<Eigen/Geometry>
#include<vector>
#include "ImuTypes.h"
#include "core/Types.h"

namespace vio{
class ImuPreintegrator{
    public:
        ImuPreintegrator(
            const ImuNoiseParams& noise,
            const Eigen::Vector3d& bias_gyro,
            const Eigen::Vector3d& bias_accel
        );
        
        // Integrate a single IMU measurement
        void integrate (
            const Eigen::Vector3d& gyro,
            const Eigen::Vector3d& accel,
            double dt
        
        );

        // First order bias correction
        PreintegrationResult getCorrected(
            const Eigen::Vector3d& new_bg,
            const Eigen::Vector3d& new_ba
        ) const ;

        // get result at current linearization point
        PreintegrationResult getResult() const;

        void reset(const Eigen::Vector3d& bias_gyro, const Eigen::Vector3d& bias_accel);
        double deltaT() const{return dt_sum_;}

    private:
        // Preinegrated measurements
        Eigen::Quaterniond dR_;
        Eigen::Vector3d dv_, dp_;

        // Covariance and bia jacobians
        Eigen::Matrix<double, 9,9> cov_;
        Eigen::Matrix<double, 9,3> J_bg_;
        Eigen::Matrix<double, 9,3> J_ba_;

        // Linearization Point
        Eigen::Vector3d bg0_, ba0_;
        ImuNoiseParams noise_;
        double dt_sum_ = 0.0;


};



}