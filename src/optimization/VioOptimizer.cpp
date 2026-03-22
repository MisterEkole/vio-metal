#include "VioOptimizer.h"
#include "optimization/Factors.h"
#include <ceres/ceres.h>
#include <unordered_set>
#include <algorithm>
#include <iostream>

namespace vio {

VioOptimizer::VioOptimizer(const Config& config, const StereoCalibration& calib)
    : config_(config), calib_(calib) {}

void VioOptimizer::initialize(const KeyframeState& initial_state) {
    window_.clear();
    preint_between_.clear();
    observations_.clear();
    landmarks_.clear();
    loop_constraints_.clear();

    window_.push_back(initial_state);
    initialized_ = true;

    std::cout << "[VioOptimizer] Initialized at t=" << initial_state.timestamp
              << " pos=(" << initial_state.position.transpose() << ")\n";
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
        observations_.erase(window_.front().timestamp);
        window_.pop_front();
        if (!preint_between_.empty()) {
            preint_between_.pop_front();
        }
    }
}

KeyframeState VioOptimizer::latestState() const {
    return window_.empty() ? KeyframeState() : window_.back();
}

// Pack keyframe state into [p(3), q_xyzw(4), v(3), bg(3), ba(3)]
static Eigen::VectorXd packState(const KeyframeState& s) {
    Eigen::VectorXd x(16);
    x.head<3>() = s.position;
    x(3) = s.rotation.x();
    x(4) = s.rotation.y();
    x(5) = s.rotation.z();
    x(6) = s.rotation.w();
    x.segment<3>(7)  = s.velocity;
    x.segment<3>(10) = s.bias_gyro;
    x.segment<3>(13) = s.bias_accel;
    return x;
}

void VioOptimizer::marginalize() {
    if (window_.size() <= 1) return;

    // Schur complement: marginalize window_[0], keep window_[1]
    const auto& state0 = window_[0];
    const auto& state1 = window_[1];

    struct MiniState {
        double p[3];
        double q[4]; // x,y,z,w
        double vba[9];
    };
    MiniState ms[2];

    auto fillMiniState = [](MiniState& ms, const KeyframeState& s) {
        Eigen::Map<Eigen::Vector3d>(ms.p) = s.position;
        ms.q[0] = s.rotation.x(); ms.q[1] = s.rotation.y();
        ms.q[2] = s.rotation.z(); ms.q[3] = s.rotation.w();
        Eigen::Map<Eigen::Vector3d>(ms.vba)     = s.velocity;
        Eigen::Map<Eigen::Vector3d>(ms.vba + 3) = s.bias_gyro;
        Eigen::Map<Eigen::Vector3d>(ms.vba + 6) = s.bias_accel;
    };
    fillMiniState(ms[0], state0);
    fillMiniState(ms[1], state1);

    ceres::Problem mini_problem;
    mini_problem.AddParameterBlock(ms[0].p, 3);
    mini_problem.AddParameterBlock(ms[0].q, 4, new ceres::EigenQuaternionManifold());
    mini_problem.AddParameterBlock(ms[0].vba, 9);
    mini_problem.AddParameterBlock(ms[1].p, 3);
    mini_problem.AddParameterBlock(ms[1].q, 4, new ceres::EigenQuaternionManifold());
    mini_problem.AddParameterBlock(ms[1].vba, 9);

    // Prior on frame 0
    if (margin_info_.hasPrior()) {
        auto* prior_cf = new ceres::AutoDiffCostFunction<
            MarginalizationFactor, 15, 3, 4, 9>(
            new MarginalizationFactor(margin_info_.sqrtInfo(),
                                      margin_info_.linearizationPoint(),
                                      margin_info_.residualOffset()));
        mini_problem.AddResidualBlock(prior_cf, nullptr, ms[0].p, ms[0].q, ms[0].vba);
    }

    // IMU factor between frame 0 and 1
    if (!preint_between_.empty() && preint_between_[0].dt > 1e-7) {
        Eigen::Vector3d g = gravity();
        auto* imu_cf = new ceres::AutoDiffCostFunction<
            ImuFactor, 15, 3, 4, 9, 3, 4, 9>(
            new ImuFactor(preint_between_[0], g));
        mini_problem.AddResidualBlock(imu_cf, nullptr,
            ms[0].p, ms[0].q, ms[0].vba,
            ms[1].p, ms[1].q, ms[1].vba);
    }

    // Reprojection factors from frame 0
    double fx = calib_.intrinsics_left[0], fy = calib_.intrinsics_left[1];
    double cx = calib_.intrinsics_left[2], cy = calib_.intrinsics_left[3];
    double baseline = calib_.T_cam1_cam0.block<3,1>(0,3).norm();
    Eigen::Matrix4d T_bc = calib_.T_cam0_imu;
    Eigen::Matrix2d sqrt_info_mono = Eigen::Matrix2d::Identity() * (1.0 / 1.5);
    Eigen::Matrix4d sqrt_info_stereo = Eigen::Matrix4d::Identity() * (1.0 / 1.5);

    auto obs_it = observations_.find(state0.timestamp);
    int num_lm = 0;
    if (obs_it != observations_.end()) {
        for (const auto& obs : obs_it->second) {
            auto lm_it = landmarks_.find(obs.feature_id);
            if (lm_it != landmarks_.end() && lm_it->second.allFinite()) num_lm++;
        }
    }

    std::vector<std::array<double, 3>> mini_lm_storage;
    mini_lm_storage.reserve(num_lm);

    if (obs_it != observations_.end()) {
        for (const auto& obs : obs_it->second) {
            auto lm_it = landmarks_.find(obs.feature_id);
            if (lm_it == landmarks_.end() || !lm_it->second.allFinite()) continue;

            mini_lm_storage.push_back({lm_it->second.x(), lm_it->second.y(), lm_it->second.z()});
            double* lm_data = mini_lm_storage.back().data();
            mini_problem.AddParameterBlock(lm_data, 3);

            if (obs.has_stereo) {
                auto* cf = new ceres::AutoDiffCostFunction<StereoReprojectionFactor, 4, 3, 4, 3>(
                    new StereoReprojectionFactor(obs.pixel_left, obs.pixel_right,
                        fx, fy, cx, cy, baseline, T_bc, sqrt_info_stereo));
                mini_problem.AddResidualBlock(cf, nullptr, ms[0].p, ms[0].q, lm_data);
            } else {
                auto* cf = new ceres::AutoDiffCostFunction<MonoReprojectionFactor, 2, 3, 4, 3>(
                    new MonoReprojectionFactor(obs.pixel_left, fx, fy, cx, cy, T_bc, sqrt_info_mono));
                mini_problem.AddResidualBlock(cf, nullptr, ms[0].p, ms[0].q, lm_data);
            }
        }
    }

    std::vector<double*> param_blocks;
    param_blocks.push_back(ms[0].p);
    param_blocks.push_back(ms[0].q);
    param_blocks.push_back(ms[0].vba);
    param_blocks.push_back(ms[1].p);
    param_blocks.push_back(ms[1].q);
    param_blocks.push_back(ms[1].vba);
    for (auto& lm : mini_lm_storage) {
        param_blocks.push_back(lm.data());
    }

    int dim_m = 15 + 3 * static_cast<int>(mini_lm_storage.size());
    int dim_k = 15;
    int dim_total = dim_m + dim_k;

    ceres::Problem::EvaluateOptions eval_opts;
    eval_opts.parameter_blocks = param_blocks;
    eval_opts.apply_loss_function = false;

    double total_cost = 0;
    std::vector<double> residuals_vec;
    ceres::CRSMatrix jacobian_crs;

    bool eval_ok = mini_problem.Evaluate(eval_opts, &total_cost, &residuals_vec, nullptr, &jacobian_crs);

    if (!eval_ok || jacobian_crs.num_cols == 0) {
        Eigen::Matrix<double, 15, 15> H_fallback = Eigen::Matrix<double, 15, 15>::Zero();
        H_fallback.block<3,3>(0,0)   = Eigen::Matrix3d::Identity() * 1e0;
        H_fallback.block<3,3>(3,3)   = Eigen::Matrix3d::Identity() * 1e0;
        H_fallback.block<3,3>(6,6)   = Eigen::Matrix3d::Identity() * 1e0;
        H_fallback.block<3,3>(9,9)   = Eigen::Matrix3d::Identity() * 1e1;
        H_fallback.block<3,3>(12,12) = Eigen::Matrix3d::Identity() * 1e1;
        Eigen::VectorXd b_fallback = Eigen::VectorXd::Zero(15);
        margin_info_.setPrior(H_fallback, b_fallback, packState(state1));
    } else {
        int num_rows = jacobian_crs.num_rows;
        int num_cols = jacobian_crs.num_cols;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_rows, num_cols);
        for (int row = 0; row < num_rows; ++row) {
            for (int idx = jacobian_crs.rows[row]; idx < jacobian_crs.rows[row + 1]; ++idx) {
                J(row, jacobian_crs.cols[idx]) = jacobian_crs.values[idx];
            }
        }
        Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(residuals_vec.data(), num_rows);

        // H = J^T J, b = -J^T r
        Eigen::MatrixXd H_full = J.transpose() * J;
        Eigen::VectorXd b_full = -J.transpose() * r;

        if (num_cols == dim_total) {
            // Reorder: [m0(15), k(15), lm(3*n)] → [m0+lm, k]
            Eigen::MatrixXd H_reordered(dim_total, dim_total);
            Eigen::VectorXd b_reordered(dim_total);

            std::vector<int> perm;
            for (int i = 0; i < 15; ++i) perm.push_back(i);
            for (int i = 30; i < dim_total; ++i) perm.push_back(i);
            for (int i = 15; i < 30; ++i) perm.push_back(i);

            for (int i = 0; i < dim_total; ++i) {
                b_reordered(i) = b_full(perm[i]);
                for (int j = 0; j < dim_total; ++j) {
                    H_reordered(i, j) = H_full(perm[i], perm[j]);
                }
            }

            // Schur complement: H_prior = H_kk - H_km * H_mm^{-1} * H_mk
            Eigen::MatrixXd H_mm = H_reordered.topLeftCorner(dim_m, dim_m);
            Eigen::MatrixXd H_mk = H_reordered.topRightCorner(dim_m, dim_k);
            Eigen::MatrixXd H_km = H_reordered.bottomLeftCorner(dim_k, dim_m);
            Eigen::MatrixXd H_kk = H_reordered.bottomRightCorner(dim_k, dim_k);
            Eigen::VectorXd b_m = b_reordered.head(dim_m);
            Eigen::VectorXd b_k = b_reordered.tail(dim_k);

            H_mm.diagonal().array() += 1e-3;
            Eigen::MatrixXd H_mm_inv = H_mm.ldlt().solve(Eigen::MatrixXd::Identity(dim_m, dim_m));
            Eigen::MatrixXd H_prior = H_kk - H_km * H_mm_inv * H_mk;
            Eigen::VectorXd b_prior = b_k - H_km * H_mm_inv * b_m;

            H_prior = 0.5 * (H_prior + H_prior.transpose());

            margin_info_.setPrior(H_prior, b_prior, packState(state1));
        } else {
            Eigen::Matrix<double, 15, 15> H_fallback = Eigen::Matrix<double, 15, 15>::Identity() * 1e0;
            margin_info_.setPrior(H_fallback, Eigen::VectorXd::Zero(15), packState(state1));
        }
    }

    // Prune loop constraints for removed frame
    uint64_t removed_ts = window_.front().timestamp;
    loop_constraints_.erase(
        std::remove_if(loop_constraints_.begin(), loop_constraints_.end(),
            [removed_ts](const LoopConstraint& lc) {
                return lc.timestamp_i == removed_ts || lc.timestamp_j == removed_ts;
            }),
        loop_constraints_.end());

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

void VioOptimizer::addLoopConstraint(const LoopConstraint& lc) {
    loop_constraints_.push_back(lc);
}

KeyframeState VioOptimizer::optimize() {
    if (window_.size() < 2) return window_.back();

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

        // Bias bounds
        for (int j = 3; j < 6; ++j) {
            problem.SetParameterLowerBound(params[i].vba, j, -0.2);
            problem.SetParameterUpperBound(params[i].vba, j,  0.2);
        }
        for (int j = 6; j < 9; ++j) {
            problem.SetParameterLowerBound(params[i].vba, j, -1.0);
            problem.SetParameterUpperBound(params[i].vba, j,  1.0);
        }
    }

    // Fix oldest frame as gauge reference
    problem.SetParameterBlockConstant(params[0].p);
    problem.SetParameterBlockConstant(params[0].q);
    problem.SetParameterBlockConstant(params[0].vba);

    // Select top landmarks by observation count
    std::unordered_map<uint64_t, std::array<double, 3>> lm_params;
    {
        std::unordered_map<uint64_t, int> obs_count;
        for (const auto& [ts, obs_list] : observations_) {
            for (const auto& obs : obs_list) {
                if (landmarks_.count(obs.feature_id)) obs_count[obs.feature_id]++;
            }
        }

        struct LmCandidate { uint64_t id; int count; };
        std::vector<LmCandidate> candidates;
        for (const auto& [id, pos] : landmarks_) {
            if (!pos.allFinite()) continue;
            if (pos.norm() > 1e4) continue;
            int cnt = obs_count.count(id) ? obs_count[id] : 0;
            if (cnt > 0) candidates.push_back({id, cnt});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const LmCandidate& a, const LmCandidate& b) { return a.count > b.count; });

        int max_lm = config_.max_landmarks;
        int num_to_use = std::min(static_cast<int>(candidates.size()), max_lm);
        for (int i = 0; i < num_to_use; ++i) {
            const auto& pos = landmarks_[candidates[i].id];
            lm_params[candidates[i].id] = {pos.x(), pos.y(), pos.z()};
            problem.AddParameterBlock(lm_params[candidates[i].id].data(), 3);
        }
    }

    // Vision residuals
    double fx = calib_.intrinsics_left[0], fy = calib_.intrinsics_left[1];
    double cx = calib_.intrinsics_left[2], cy = calib_.intrinsics_left[3];
    double baseline = calib_.T_cam1_cam0.block<3,1>(0,3).norm();
    Eigen::Matrix4d T_bc = calib_.T_cam0_imu;
    Eigen::Matrix3d R_bc = T_bc.block<3,3>(0,0);
    Eigen::Vector3d t_bc = T_bc.block<3,1>(0,3);

    Eigen::Matrix2d sqrt_info_mono = Eigen::Matrix2d::Identity() * (1.0 / 1.5);
    Eigen::Matrix4d sqrt_info_stereo = Eigen::Matrix4d::Identity() * (1.0 / 1.5);

    // Depth filter: skip if too few landmarks pass
    int depth_pass_count = 0;
    for (int i = 0; i < n; ++i) {
        auto obs_it = observations_.find(window_[i].timestamp);
        if (obs_it == observations_.end()) continue;
        const auto& state = window_[i];
        for (const auto& obs : obs_it->second) {
            auto lm_it = lm_params.find(obs.feature_id);
            if (lm_it == lm_params.end()) continue;
            Eigen::Vector3d p_w(lm_it->second[0], lm_it->second[1], lm_it->second[2]);
            Eigen::Vector3d p_b = state.rotation.inverse() * (p_w - state.position);
            Eigen::Vector3d p_c = R_bc * p_b + t_bc;
            if (p_c.z() >= 0.1 && p_c.z() <= 200.0) depth_pass_count++;
        }
    }

    bool use_depth_filter = (depth_pass_count >= 20);

    int vision_residual_count = 0;
    for (int i = 0; i < n; ++i) {
        auto obs_it = observations_.find(window_[i].timestamp);
        if (obs_it == observations_.end()) continue;

        const auto& state = window_[i];
        for (const auto& obs : obs_it->second) {
            auto lm_it = lm_params.find(obs.feature_id);
            if (lm_it == lm_params.end()) continue;

            if (use_depth_filter) {
                Eigen::Vector3d p_w(lm_it->second[0], lm_it->second[1], lm_it->second[2]);
                Eigen::Vector3d p_b = state.rotation.inverse() * (p_w - state.position);
                Eigen::Vector3d p_c = R_bc * p_b + t_bc;
                if (p_c.z() < 0.1 || p_c.z() > 200.0) continue;
            }

            auto* loss = new ceres::HuberLoss(config_.huber_reprojection);
            if (obs.has_stereo) {
                auto* cf = new ceres::AutoDiffCostFunction<StereoReprojectionFactor, 4, 3, 4, 3>(
                    new StereoReprojectionFactor(obs.pixel_left, obs.pixel_right, fx, fy, cx, cy, baseline, T_bc, sqrt_info_stereo));
                problem.AddResidualBlock(cf, loss, params[i].p, params[i].q, lm_it->second.data());
            } else {
                auto* cf = new ceres::AutoDiffCostFunction<MonoReprojectionFactor, 2, 3, 4, 3>(
                    new MonoReprojectionFactor(obs.pixel_left, fx, fy, cx, cy, T_bc, sqrt_info_mono));
                problem.AddResidualBlock(cf, loss, params[i].p, params[i].q, lm_it->second.data());
            }
            vision_residual_count++;
        }
    }

    // IMU residuals
    Eigen::Vector3d g = gravity();
    for (int i = 0; i < n - 1; ++i) {
        if (i >= static_cast<int>(preint_between_.size())) break;
        if (preint_between_[i].dt < 1e-7) continue;

        auto* imu_cf = new ceres::AutoDiffCostFunction<ImuFactor, 15, 3, 4, 9, 3, 4, 9>(
            new ImuFactor(preint_between_[i], g));

        problem.AddResidualBlock(imu_cf, new ceres::HuberLoss(10.0),
                                 params[i].p, params[i].q, params[i].vba,
                                 params[i+1].p, params[i+1].q, params[i+1].vba);
    }

    // Loop closure residuals
    for (const auto& lc : loop_constraints_) {
        int idx_i = -1, idx_j = -1;
        for (int i = 0; i < n; ++i) {
            if (window_[i].timestamp == lc.timestamp_i) idx_i = i;
            if (window_[i].timestamp == lc.timestamp_j) idx_j = i;
        }
        if (idx_i >= 0 && idx_j >= 0) {
            auto* cf = new ceres::AutoDiffCostFunction<LoopClosureFactor, 6, 3, 4, 3, 4>(
                new LoopClosureFactor(lc.relative_position, lc.relative_rotation, lc.sqrt_info));
            problem.AddResidualBlock(cf, new ceres::HuberLoss(1.0),
                params[idx_i].p, params[idx_i].q, params[idx_j].p, params[idx_j].q);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = config_.max_iterations;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;
    if (config_.use_dogleg) {
        options.trust_region_strategy_type = ceres::DOGLEG;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    last_summary_.initial_cost = summary.initial_cost;
    last_summary_.final_cost = summary.final_cost;
    last_summary_.iterations = static_cast<int>(summary.iterations.size());
    last_summary_.num_residuals = summary.num_residuals;
    last_summary_.num_landmarks = static_cast<int>(lm_params.size());
    last_summary_.success = (summary.termination_type == ceres::CONVERGENCE);

    {
        Eigen::Vector3d optimized_pos = Eigen::Map<Eigen::Vector3d>(params[n-1].p);
        Eigen::Vector3d pre_opt_pos = window_.back().position;
        double jump = (optimized_pos - pre_opt_pos).norm();
        if (jump > 1.0) {
            std::printf("[VioOptimizer] correction: %.2fm\n", jump);
        }
    }

    // Protect biases if too few visual constraints
    bool trust_vision = (vision_residual_count >= 20);

    for (int i = 0; i < n; ++i) {
        auto& state = window_[i];
        state.position = Eigen::Map<Eigen::Vector3d>(params[i].p);
        state.rotation.x() = params[i].q[0];
        state.rotation.y() = params[i].q[1];
        state.rotation.z() = params[i].q[2];
        state.rotation.w() = params[i].q[3];
        state.rotation.normalize();

        state.velocity   = Eigen::Map<Eigen::Vector3d>(params[i].vba);

        if (trust_vision) {
            state.bias_gyro  = Eigen::Map<Eigen::Vector3d>(params[i].vba + 3);
            state.bias_accel = Eigen::Map<Eigen::Vector3d>(params[i].vba + 6);
        }
    }

    // Write back optimized landmarks
    for (auto& [id, pos] : landmarks_) {
        auto it = lm_params.find(id);
        if (it != lm_params.end()) {
            pos = Eigen::Vector3d(it->second[0], it->second[1], it->second[2]);
        }
    }

    // Prune landmarks with >15px reprojection error or behind camera
    {
        const auto& latest = window_.back();
        std::vector<uint64_t> to_remove;
        for (const auto& [id, pos] : landmarks_) {
            if (!pos.allFinite()) { to_remove.push_back(id); continue; }
            Eigen::Vector3d p_b = latest.rotation.inverse() * (pos - latest.position);
            Eigen::Vector3d p_c = R_bc * p_b + t_bc;
            if (p_c.z() < 0.01) { to_remove.push_back(id); continue; }
            auto obs_it = observations_.find(latest.timestamp);
            if (obs_it == observations_.end()) continue;
            for (const auto& obs : obs_it->second) {
                if (obs.feature_id != id) continue;
                double u = fx * p_c.x() / p_c.z() + cx;
                double v = fy * p_c.y() / p_c.z() + cy;
                double err = std::sqrt((u - obs.pixel_left.x()) * (u - obs.pixel_left.x())
                                     + (v - obs.pixel_left.y()) * (v - obs.pixel_left.y()));
                if (err > 15.0) to_remove.push_back(id);
                break;
            }
        }
        for (uint64_t id : to_remove) {
            landmarks_.erase(id);
        }
    }

    return window_.back();
}

}
