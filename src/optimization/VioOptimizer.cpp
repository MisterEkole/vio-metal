#include "VioOptimizer.h"
#include "optimization/Factors.h"
#include <ceres/ceres.h>
#include <ceres/numeric_diff_cost_function.h>
#include <unordered_set>
#include <iostream>

namespace vio {

VioOptimizer::VioOptimizer(const Config& config, const StereoCalibration& calib)
    : config_(config), calib_(calib) {}

void VioOptimizer::initialize(const KeyframeState& initial_state) {
    window_.clear();
    preint_between_.clear();
    observations_.clear();
    landmarks_.clear();

    window_.push_back(initial_state);
    initialized_ = true;

    std::cout << "[VioOptimizer] Initialized at t=" << initial_state.timestamp
              << " pos=(" << initial_state.position.transpose() << ")\n";
    std::cout << "Extrinsic Translation (IMU to Cam): " << calib_.T_cam0_imu.block<3,1>(0,3).transpose() << std::endl;
}

void VioOptimizer::addKeyframe(
    uint64_t timestamp,
    const PreintegrationResult& preint,
    const std::vector<FeatureObservation>& observations)
{
    if (!initialized_) {
        std::cerr << "[VioOptimizer] ERROR: addKeyframe called before initialize!\n";
        return;
    }

    const auto& prev = window_.back();
    KeyframeState new_kf;
    new_kf.timestamp = timestamp;

    Eigen::Matrix3d R_prev = prev.rotation.toRotationMatrix();
    Eigen::Vector3d g = gravity();
    double dt = preint.dt;

    // IMU-based prediction for the next state
    new_kf.rotation = prev.rotation * preint.delta_R;
    new_kf.rotation.normalize();
    new_kf.velocity = prev.velocity + g * dt + R_prev * preint.delta_v;
    new_kf.position = prev.position + prev.velocity * dt
                    + 0.5 * g * dt * dt + R_prev * preint.delta_p;
    new_kf.bias_gyro = prev.bias_gyro;
    new_kf.bias_accel = prev.bias_accel;

    window_.push_back(new_kf);
    preint_between_.push_back(preint);
    observations_[timestamp] = observations;

    if (static_cast<int>(window_.size()) > config_.window_size) {
        marginalize();
    }
}

KeyframeState VioOptimizer::latestState() const {
    return window_.empty() ? KeyframeState() : window_.back();
}

void VioOptimizer::marginalize() {
    if (window_.size() <= 1) return;
    observations_.erase(window_.front().timestamp);
    window_.pop_front();
    if (!preint_between_.empty()) {
        preint_between_.pop_front();
    }
}

void VioOptimizer::setLandmarks(
    const std::unordered_map<uint64_t, Eigen::Vector3d>& landmarks)
{
    for (const auto& [id, pos] : landmarks) {
        if (landmarks_.find(id) == landmarks_.end()) {
            if (pos.allFinite()) {
                landmarks_[id] = pos;
            }
        }
    }
}

void VioOptimizer::addObservations(
    uint64_t timestamp, const std::vector<FeatureObservation>& obs)
{
    observations_[timestamp] = obs;
}

KeyframeState VioOptimizer::optimize() {
    if (window_.size() < 2) return window_.back();

    // Prune stale landmarks
    {
        std::unordered_set<uint64_t> visible_ids;
        for (const auto& [ts, obs_list] : observations_) {
            for (const auto& obs : obs_list) visible_ids.insert(obs.feature_id);
        }
        for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
            if (visible_ids.find(it->first) == visible_ids.end()) it = landmarks_.erase(it);
            else ++it;
        }
    }

    ceres::Problem problem;
    int n = static_cast<int>(window_.size());

    struct LocalState {
        double p[3];
        double q[4];   // x, y, z, w
        double vba[9]; // vel(3), bg(3), ba(3)
    };
    auto params = std::make_unique<LocalState[]>(n); 
   

    // Add Parameter Blocks
    for (int i = 0; i < n; ++i) {
        const auto& state = window_[i];
        Eigen::Map<Eigen::Vector3d>(params[i].p) = state.position;
        params[i].q[0] = state.rotation.x();
        params[i].q[1] = state.rotation.y();
        params[i].q[2] = state.rotation.z();
        params[i].q[3] = state.rotation.w();
        Eigen::Map<Eigen::Vector3d>(params[i].vba)     = state.velocity;
        Eigen::Map<Eigen::Vector3d>(params[i].vba + 3) = state.bias_gyro;
        Eigen::Map<Eigen::Vector3d>(params[i].vba + 6) = state.bias_accel;

        problem.AddParameterBlock(params[i].p, 3);
        problem.AddParameterBlock(params[i].q, 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(params[i].vba, 9);
    }

    // Fix first frame to anchor the map
    problem.SetParameterBlockConstant(params[0].p);
    problem.SetParameterBlockConstant(params[0].q);
    problem.SetParameterBlockConstant(params[0].vba);

    // Landmark parameters
    std::unordered_map<uint64_t, std::array<double, 3>> lm_params;
    for (const auto& [id, pos] : landmarks_) {
        if (!pos.allFinite()) continue;
        lm_params[id] = {pos.x(), pos.y(), pos.z()};
        problem.AddParameterBlock(lm_params[id].data(), 3);
    }

    // Vision Residuals
    double fx = calib_.intrinsics_left[0], fy = calib_.intrinsics_left[1];
    double cx = calib_.intrinsics_left[2], cy = calib_.intrinsics_left[3];
    double baseline = calib_.T_cam1_cam0.block<3,1>(0,3).norm();
    Eigen::Matrix4d T_bc = calib_.T_cam0_imu; 

    // Weighting: Inverse of pixel noise (e.g., 1.5 pixel std dev)
    Eigen::Matrix2d sqrt_info_mono = Eigen::Matrix2d::Identity() * (1.0 / 1.5);
    Eigen::Matrix4d sqrt_info_stereo = Eigen::Matrix4d::Identity() * (1.0 / 1.5);

    int num_stereo_res = 0, num_mono_res = 0;
    for (int i = 0; i < n; ++i) {
        auto obs_it = observations_.find(window_[i].timestamp);
        if (obs_it == observations_.end()) continue;

        for (const auto& obs : obs_it->second) {
            auto lm_it = lm_params.find(obs.feature_id);
            if (lm_it == lm_params.end()) continue;

            auto* loss = new ceres::HuberLoss(config_.huber_reprojection);
            if (obs.has_stereo) {
                auto* cf = new ceres::AutoDiffCostFunction<StereoReprojectionFactor, 4, 3, 4, 3>(
                    new StereoReprojectionFactor(obs.pixel_left, obs.pixel_right, fx, fy, cx, cy, baseline, T_bc, sqrt_info_stereo));
                problem.AddResidualBlock(cf, loss, params[i].p, params[i].q, lm_it->second.data());
                num_stereo_res++;
            } else {
                auto* cf = new ceres::AutoDiffCostFunction<MonoReprojectionFactor, 2, 3, 4, 3>(
                    new MonoReprojectionFactor(obs.pixel_left, fx, fy, cx, cy, T_bc, sqrt_info_mono));
                problem.AddResidualBlock(cf, loss, params[i].p, params[i].q, lm_it->second.data());
                num_mono_res++;
            }
        }
    }

    // IMU Residuals 
    Eigen::Vector3d g = gravity();
    for (int i = 0; i < n - 1; ++i) {
        if (i >= static_cast<int>(preint_between_.size())) break;
        if (preint_between_[i].dt < 1e-7) continue;

        auto* imu_cf = new ceres::NumericDiffCostFunction<ImuFactor, ceres::CENTRAL, 15, 3, 4, 9, 3, 4, 9>(
            new ImuFactor(preint_between_[i], g));
        
        problem.AddResidualBlock(imu_cf, new ceres::HuberLoss(10.0),
                                 params[i].p, params[i].q, params[i].vba,
                                 params[i+1].p, params[i+1].q, params[i+1].vba);
    }

    // --- Debug Code ---
    for (int i = 0; i < n - 1; ++i) {
        if (i >= static_cast<int>(preint_between_.size())) break;
        double res[15];
        ImuFactor factor(preint_between_[i], g);
        factor(params[i].p, params[i].q, params[i].vba, 
               params[i+1].p, params[i+1].q, params[i+1].vba, res);
        Eigen::Map<Eigen::Matrix<double, 15, 1>> res_vec(res);
        std::cout << "[IMU Check #" << i << "] Initial Residual Norm: " << res_vec.norm() << std::endl;
    }

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = config_.max_iterations;
    options.num_threads = 8; 
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Write-back results to window
    for (int i = 0; i < n; ++i) {
        auto& state = window_[i];
        state.position = Eigen::Map<Eigen::Vector3d>(params[i].p);
        state.rotation.x() = params[i].q[0];
        state.rotation.y() = params[i].q[1];
        state.rotation.z() = params[i].q[2];
        state.rotation.w() = params[i].q[3];
        state.rotation.normalize();
    
        state.velocity   = Eigen::Map<Eigen::Vector3d>(params[i].vba);
        state.bias_gyro  = Eigen::Map<Eigen::Vector3d>(params[i].vba + 3);
        state.bias_accel = Eigen::Map<Eigen::Vector3d>(params[i].vba + 6);
    }

    // Update global landmark map
    for (auto& [id, pos] : landmarks_) {
        auto it = lm_params.find(id);
        if (it != lm_params.end()) {
            pos = Eigen::Vector3d(it->second[0], it->second[1], it->second[2]);
        }
    }

    return window_.back();
}

} 