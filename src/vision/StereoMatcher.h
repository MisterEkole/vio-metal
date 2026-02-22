#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include "dataset/EurocLoader.h"

namespace vio {

class StereoMatcher {
public:
    struct Config {
        double max_epipolar_error = 2.0;    // pixels (vertical disparity after rectification)
        double max_descriptor_dist = 50.0;  // Hamming distance for ORB
        double min_disparity = 1.0;         // pixels (max depth cutoff)
        double max_disparity = 120.0;       // pixels (min depth cutoff)
        double ratio_test = 0.8;            // Lowe's ratio test
    };

    struct StereoMatch {
        int left_idx;
        int right_idx;
        double disparity;
        Eigen::Vector3d point_3d;  // Triangulated in left camera frame
    };
    StereoMatcher();

    explicit StereoMatcher(const Config& config);

    std::vector<StereoMatch> match(
        const std::vector<cv::KeyPoint>& left_kpts,
        const cv::Mat& left_desc,
        const std::vector<cv::KeyPoint>& right_kpts,
        const cv::Mat& right_desc,
        const StereoCalibration& calib
    ) const;

private:
    Config config_;
};

} 
