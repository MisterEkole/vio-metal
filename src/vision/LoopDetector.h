#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace vio {

struct KeyframeDescriptorEntry {
    uint64_t timestamp;
    cv::Mat descriptors;                    // ORB descriptors (N x 32, CV_8U)
    std::vector<cv::KeyPoint> keypoints;
    std::vector<uint64_t> feature_ids;
    std::unordered_map<uint64_t, Eigen::Vector3d> landmarks_world;
};

struct LoopCandidate {
    uint64_t query_timestamp = 0;
    uint64_t match_timestamp = 0;
    Eigen::Vector3d relative_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond relative_rotation = Eigen::Quaterniond::Identity();
    int num_inliers = 0;
    bool valid = false;
};

class LoopDetector {
public:
    struct Config {
        int min_loop_gap;
        int min_descriptor_matches;
        double max_hamming_ratio;
        int ransac_iterations;
        double ransac_reproj_thresh;
        int min_pnp_inliers;

        Config()
            : min_loop_gap(10)
            , min_descriptor_matches(25)
            , max_hamming_ratio(0.75)
            , ransac_iterations(200)
            , ransac_reproj_thresh(5.0)
            , min_pnp_inliers(12) {}
    };

    explicit LoopDetector(const Config& config = Config());

    void addKeyframe(const KeyframeDescriptorEntry& entry);

    LoopCandidate detectLoop(
        const KeyframeDescriptorEntry& query,
        double fx, double fy, double cx, double cy,
        const Eigen::Matrix4d& T_cam_imu);

    int databaseSize() const { return static_cast<int>(database_.size()); }

private:
    std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat& query_desc, const cv::Mat& candidate_desc);

    Config config_;
    std::vector<KeyframeDescriptorEntry> database_;
    cv::BFMatcher matcher_;
};

} // namespace vio
