#include "ImuPreintegrator.h"
#include <iostream>

namespace vio{
    ImuPreintegrator::ImuPreintegrator(
        const ImuNoiseParams& noise,
        const Eigen::Vector3d& bias_gyro,
        const Eigen::Vector3d& bias_accel
    ) : noise_(noise), bg0_(bias_gyro), ba0_(bias_accel){
        reset(bias_gyro, bias_accel);
    }

    void ImuPreintegrator::reset(const Eigen::Vector3d& bias_gyro, const Eigen::Vector3d& bias_accel){
        dR_ = Eigen::Quaterniond::Identity();
        dv_ = Eigen::Vector3d::Zero();
        dp_ = Eigen::Vector3d::Zero();

        cov_.setZero();
        J_bg_.setZero();
        J_ba_.setZero();
        bg0_= bias_gyro;
        ba0_= bias_accel;
        dt_sum_ = 0.0;
    }

    void ImuPreintegrator::integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt){
        // Bias corrected measurements

        Eigen::Vector3d omega = gyro - bg0_;
        Eigen::Vector3d acc = accel - ba0_;

        Eigen::Matrix3d R_k = dR_.toRotationMatrix();

        // Rotation increment
        Eigen::Vector3d omega_dt = omega * dt;
        Eigen::Matrix3d dR_inc = expSO3(omega_dt);
        Eigen::Matrix3d Jr = rightJacobianSO3(omega_dt);

        // Covariance propagation
        Eigen::Matrix<double, 9,9> A = Eigen::Matrix<double,9,9>::Identity();
        Eigen::Matrix<double, 9,6> B = Eigen::Matrix<double,9,6>::Zero();

        // A matrix (error-state transition)
        A.block<3,3>(0,0) = dR_inc.transpose();
        A.block<3,3>(3,0) = -R_k * skewSymmetric(acc) * dt;
        A.block<3,3>(6,0) = -0.5 * R_k * skewSymmetric(acc) * dt * dt;
        A.block<3,3>(6,3) = Eigen::Matrix3d::Identity() * dt;  // BUG FIX 1: was missing

        // B matrix (noise input: maps measurement noise to state error)
        B.block<3,3>(0,0) = Jr * dt;                   // Gyro noise → rotation error
        B.block<3,3>(3,3) = R_k * dt;                  // BUG FIX 2: was -R_k*skew(acc)*dt (an A-matrix term!)
        B.block<3,3>(6,3) = 0.5 * R_k * dt * dt;      // BUG FIX 3: was I*dt

        // Noise covariance (continuous PSD → discrete)
        Eigen::Matrix<double, 6,6> Qc = Eigen::Matrix<double, 6,6>::Zero();
        double ng2 = noise_.gyro_noise_density * noise_.gyro_noise_density;   // BUG FIX 4: must square σ
        double na2 = noise_.accel_noise_density * noise_.accel_noise_density;
        Qc.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * ng2;
        Qc.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * na2;

        cov_ = A * cov_ * A.transpose() + B * Qc * B.transpose();

        // ---- Bias Jacobians ----
        // J_bg: how preintegrated measurements change with gyro bias
        J_bg_.block<3,3>(0,0) = dR_inc.transpose() * J_bg_.block<3,3>(0,0) - Jr * dt;
        J_bg_.block<3,3>(3,0) = J_bg_.block<3,3>(3,0) - R_k * skewSymmetric(acc) * J_bg_.block<3,3>(0,0) * dt;
        J_bg_.block<3,3>(6,0) = J_bg_.block<3,3>(6,0) + J_bg_.block<3,3>(3,0) * dt
                            - 0.5 * R_k * skewSymmetric(acc) * J_bg_.block<3,3>(0,0) * dt * dt;

        // J_ba: how preintegrated measurements change with accel bias
        J_ba_.block<3,3>(3,0) = J_ba_.block<3,3>(3,0) - R_k * dt;
        J_ba_.block<3,3>(6,0) = J_ba_.block<3,3>(6,0) + J_ba_.block<3,3>(3,0) * dt - 0.5 * R_k * dt * dt;

        // ---- Update preintegrated measurements ----
        dp_ = dp_ + dv_ * dt + 0.5 * R_k * acc * dt * dt;
        dv_ = dv_ + R_k * acc * dt;
        dR_ = Eigen::Quaterniond(R_k * dR_inc);
        dR_.normalize();

         dt_sum_ += dt; 
        }
    PreintegrationResult ImuPreintegrator::getResult() const{
        PreintegrationResult r;
        r.delta_R = dR_;
        r.delta_v = dv_;
        r.delta_p = dp_;
        r.covariance = cov_;
        r.J_bg = J_bg_;
        r.J_ba = J_ba_;
        r.dt = dt_sum_;
        r.linearized_bg = bg0_;
        r.linearized_ba = ba0_;
        return r;
    

    }
    PreintegrationResult ImuPreintegrator::getCorrected(const Eigen::Vector3d& new_bg, const Eigen::Vector3d& new_ba) const{
        PreintegrationResult r = getResult();
        //bias correction
        Eigen::Vector3d dbg = new_bg - bg0_;
        Eigen::Vector3d dba = new_ba - ba0_;

        // rotation correction
        Eigen::Vector3d dR_correction = J_bg_.block<3,3>(0,0).transpose() * dbg;
        r.delta_R = dR_ * Eigen::Quaterniond(expSO3(dR_correction));
        r.delta_R.normalize();

        // vel and pos correction (ADD to original, don't overwrite)
        r.delta_v = dv_ + J_bg_.block<3,3>(3,0) * dbg + J_ba_.block<3,3>(3,0) * dba;
        r.delta_p = dp_ + J_bg_.block<3,3>(6,0) * dbg + J_ba_.block<3,3>(6,0) * dba;

        r.linearized_bg = new_bg;
        r.linearized_ba = new_ba;

        return r;  
    
    }

    
}