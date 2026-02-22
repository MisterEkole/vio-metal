#include "StereoMatcher.h"
#include <algorithm>
#include <limits>

namespace vio {
StereoMatcher::StereoMatcher() 
    : StereoMatcher(Config{}) 
{}

StereoMatcher::StereoMatcher(const Config& config)
    : config_(config) {}

std::vector<StereoMatcher::StereoMatch> StereoMatcher::match(
    const std::vector<cv::KeyPoint>& left_kpts,
    const cv::Mat& left_desc,
    const std::vector<cv::KeyPoint>& right_kpts,
    const cv::Mat& right_desc,
    const StereoCalibration& calib) const
{
    std::vector<StereoMatch> matches;
    if (left_kpts.empty() || right_kpts.empty()) return matches;

    double fx = calib.intrinsics_left[0];
    double fy = calib.intrinsics_left[1];
    double cx = calib.intrinsics_left[2];
    double cy = calib.intrinsics_left[3];

    // Baseline: distance between cameras along x-axis
    // T_cam1_cam0 column 3 gives translation of cam0 origin in cam1 frame
    Eigen::Vector3d t_cam1_cam0 = calib.T_cam1_cam0.block<3,1>(0,3);
    double baseline = t_cam1_cam0.norm();

    // For each left keypoint, find best matching right keypoint
    for (int i = 0; i < (int)left_kpts.size(); i++) {
        const auto& lkp = left_kpts[i];
        const cv::Mat& ldesc = left_desc.row(i);

        int best_idx = -1;
        int best_dist = std::numeric_limits<int>::max();
        int second_dist = std::numeric_limits<int>::max();

        for (int j = 0; j < (int)right_kpts.size(); j++) {
            const auto& rkp = right_kpts[j];

            // Epipolar constraint: for rectified stereo, y-coords should match
            double y_diff = std::abs(lkp.pt.y - rkp.pt.y);
            if (y_diff > config_.max_epipolar_error) continue;

            // Disparity constraint: right point should be to the left of (or at) left point
            double disparity = lkp.pt.x - rkp.pt.x;
            if (disparity < config_.min_disparity || disparity > config_.max_disparity) continue;

            // Descriptor distance (Hamming for ORB)
            int dist = cv::norm(ldesc, right_desc.row(j), cv::NORM_HAMMING);
            if (dist < best_dist) {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }

        // Apply thresholds
        if (best_idx < 0) continue;
        if (best_dist > config_.max_descriptor_dist) continue;

        // Ratio test
        if (second_dist < std::numeric_limits<int>::max()) {
            double ratio = (double)best_dist / (double)second_dist;
            if (ratio > config_.ratio_test) continue;
        }

        // Triangulate
        double disparity = lkp.pt.x - right_kpts[best_idx].pt.x;
        double depth = fx * baseline / disparity;

        StereoMatch m;
        m.left_idx = i;
        m.right_idx = best_idx;
        m.disparity = disparity;
        m.point_3d = Eigen::Vector3d(
            depth * (lkp.pt.x - cx) / fx,
            depth * (lkp.pt.y - cy) / fy,
            depth
        );

        matches.push_back(m);
    }

    return matches;
}

} 
