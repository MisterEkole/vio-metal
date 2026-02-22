#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "dataset/EurocLoader.h"
#include "imu/ImuPreintegrator.h"
#include "imu/ImuTypes.h"
#include "core/Types.h"
#include "optimization/Marginalization.h"

namespace vio {

struct KeyframeState {
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bias_gyro;
    Eigen::Vector3d bias_accel;
};

class VioOptimizer {
public:
    struct Config {
        int window_size = 12;
        int max_iterations = 5;
        double huber_reprojection = 1.5;
        double huber_imu = 1.0;
        bool use_dogleg = true;
    };

    VioOptimizer(const Config& config, const StereoCalibration& calib);

    // Add a new keyframe with its preintegration and observations
    void addKeyframe(
        uint64_t timestamp,
        const PreintegrationResult& preint,
        const std::vector<FeatureObservation>& observations);

    // Initialize the first keyframe (from ground truth or identity)
    void initialize(const KeyframeState& initial_state);

    // Run sliding window optimization
    KeyframeState optimize();

    // Accessors
    const std::deque<KeyframeState>& window() const { return window_; }
    KeyframeState latestState() const;
    bool isInitialized() const { return initialized_; }
    int windowSize() const { return static_cast<int>(window_.size()); }

    void setLandmarks(const std::unordered_map<uint64_t, Eigen::Vector3d>& landmarks);

private:
    void marginalize();

    Config config_;
    StereoCalibration calib_;
    bool initialized_ = false;

    // Sliding window of keyframe states
    std::deque<KeyframeState> window_;
    std::deque<PreintegrationResult> preint_between_;  // preint_between_[i] connects window_[i] to window_[i+1]

    // Observations per keyframe (indexed by timestamp)
    std::unordered_map<uint64_t, std::vector<FeatureObservation>> observations_;

    // 3D landmarks (feature_id -> world position)
    std::unordered_map<uint64_t, Eigen::Vector3d> landmarks_;

    // Marginalization prior
    MarginalizationInfo margin_info_;
};


} 
