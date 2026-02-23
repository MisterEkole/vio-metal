#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "core/Types.h"
#include "vision/StereoMatcher.h"

namespace vio {

struct FeatureTrack {
    uint64_t feature_id;
    std::vector<uint64_t> frame_timestamps;
    std::vector<Eigen::Vector2d> observations_left;
    std::vector<Eigen::Vector2d> observations_right;  // (-1,-1) if no stereo
    Eigen::Vector3d landmark_3d;         // In camera frame at first observation
    bool landmark_initialized = false;
    int num_optimized = 0;
};

class FeatureManager {
public:
    FeatureManager() = default;

    void addNewFeatures(
        uint64_t timestamp,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& descriptors,
        const std::vector<StereoMatcher::StereoMatch>& stereo_matches);

    void updateTracks(
        uint64_t timestamp,
        const std::vector<uint64_t>& tracked_ids,
        const std::vector<cv::Point2f>& tracked_points,
        const std::vector<bool>& status);

    // Update stereo observations for tracked features at keyframes
    void updateStereoForTracked(
        uint64_t timestamp,
        const std::vector<uint64_t>& tracked_ids,
        const std::vector<StereoMatcher::StereoMatch>& stereo_matches);

    // --- NEW METHOD FOR COORDINATE TRANSFORMATION ---
    // Converts local landmark_3d into World coordinates based on current pose
    std::unordered_map<uint64_t, Eigen::Vector3d> getInitializedLandmarksWorld(
        const Eigen::Vector3d& p_wb, 
        const Eigen::Quaterniond& q_wb, 
        const Eigen::Matrix4d& T_cam_imu) const;

    std::vector<cv::Point2f> getCurrentPoints() const;
    std::vector<uint64_t> getCurrentIds() const;

    std::vector<FeatureObservation> getObservationsForFrame(uint64_t timestamp) const;

    std::vector<const FeatureTrack*> getActiveTracksInWindow(
        const std::vector<uint64_t>& window_timestamps) const;

    void pruneDeadTracks(int min_track_length = 3);
    void removeTracksOlderThan(uint64_t timestamp);

    int numActiveTracks() const { return static_cast<int>(current_ids_.size()); }
    int numTotalTracks() const { return static_cast<int>(tracks_.size()); }

private:
    uint64_t next_feature_id_ = 0;
    std::unordered_map<uint64_t, FeatureTrack> tracks_;

    std::vector<uint64_t> current_ids_;
    std::vector<cv::Point2f> current_points_;
};

}