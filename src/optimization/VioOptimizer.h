#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "dataset/EurocLoader.h"
#include "imu/ImuPreintegrator.h"
#include "imu/ImuTypes.h"
#include "core/Types.h"
#include "optimization/Marginalization.h"

namespace vio {

struct KeyframeState {
    uint64_t timestamp = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();
};

struct LoopConstraint {
    uint64_t timestamp_i;               // query keyframe
    uint64_t timestamp_j;               // matched keyframe
    Eigen::Vector3d relative_position;  // T_ij translation (in frame i)
    Eigen::Quaterniond relative_rotation; // T_ij rotation
    Eigen::Matrix<double, 6, 6> sqrt_info;
};

class VioOptimizer {
public:
    struct Config {
        int window_size = 20;
        int max_iterations = 100;
        double huber_reprojection = 5.0;
        double huber_imu = 10.0;
        bool use_dogleg = false;
        int max_landmarks = 250;
        double max_position_jump = 2.0;   // meters
    };

    struct OptimizeSummary {
        double initial_cost = 0.0;
        double final_cost = 0.0;
        int iterations = 0;
        int num_residuals = 0;
        int num_landmarks = 0;
        bool success = false;
    };

    VioOptimizer(const Config& config, const StereoCalibration& calib);

    void addKeyframe(
        uint64_t timestamp,
        const PreintegrationResult& preint,
        const std::vector<FeatureObservation>& observations);

    void initialize(const KeyframeState& initial_state);
    KeyframeState optimize();
    const OptimizeSummary& lastSummary() const { return last_summary_; }

    const std::deque<KeyframeState>& window() const { return window_; }
    KeyframeState latestState() const;
    bool isInitialized() const { return initialized_; }
    int windowSize() const { return static_cast<int>(window_.size()); }

    void setLandmarks(const std::unordered_map<uint64_t, Eigen::Vector3d>& landmarks);
    void addObservations(uint64_t timestamp, const std::vector<FeatureObservation>& obs);

    void addLoopConstraint(const LoopConstraint& lc);

private:
    void marginalize();

    Config config_;
    StereoCalibration calib_;
    bool initialized_ = false;

    std::deque<KeyframeState> window_;
    std::deque<PreintegrationResult> preint_between_;

    std::unordered_map<uint64_t, std::vector<FeatureObservation>> observations_;
    std::unordered_map<uint64_t, Eigen::Vector3d> landmarks_;

    MarginalizationInfo margin_info_;
    std::vector<LoopConstraint> loop_constraints_;
    OptimizeSummary last_summary_;
};

}
