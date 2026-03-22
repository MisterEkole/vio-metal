#include "LoopDetector.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace vio {

LoopDetector::LoopDetector(const Config& config)
    : config_(config), matcher_(cv::NORM_HAMMING) {}

void LoopDetector::addKeyframe(const KeyframeDescriptorEntry& entry) {
    if (entry.descriptors.empty()) return;
    database_.push_back(entry);
}

std::vector<cv::DMatch> LoopDetector::matchDescriptors(
    const cv::Mat& query_desc, const cv::Mat& candidate_desc)
{
    if (query_desc.empty() || candidate_desc.empty()) return {};
    if (query_desc.rows < 2 || candidate_desc.rows < 2) return {};

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_.knnMatch(query_desc, candidate_desc, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < config_.max_hamming_ratio * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    return good_matches;
}

LoopCandidate LoopDetector::detectLoop(
    const KeyframeDescriptorEntry& query,
    double fx, double fy, double cx, double cy,
    const Eigen::Matrix4d& T_cam_imu)
{
    LoopCandidate result;
    if (query.descriptors.empty()) return result;

    int db_size = static_cast<int>(database_.size());
    int best_idx = -1;
    int best_match_count = 0;
    std::vector<cv::DMatch> best_matches;

    for (int c = 0; c < db_size - config_.min_loop_gap; ++c) {
        auto& candidate = database_[c];
        auto matches = matchDescriptors(query.descriptors, candidate.descriptors);
        if (static_cast<int>(matches.size()) > best_match_count &&
            static_cast<int>(matches.size()) >= config_.min_descriptor_matches) {
            best_match_count = static_cast<int>(matches.size());
            best_idx = c;
            best_matches = matches;
        }
    }

    if (best_idx < 0) return result;

    // 3D-2D correspondences for PnP
    const auto& candidate = database_[best_idx];
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;

    for (const auto& m : best_matches) {
        int query_kp_idx = m.queryIdx;
        int cand_kp_idx = m.trainIdx;

        if (cand_kp_idx >= static_cast<int>(candidate.feature_ids.size())) continue;
        uint64_t cand_fid = candidate.feature_ids[cand_kp_idx];

        auto lm_it = candidate.landmarks_world.find(cand_fid);
        if (lm_it == candidate.landmarks_world.end()) continue;

        const auto& pw = lm_it->second;
        if (!pw.allFinite()) continue;

        pts_3d.emplace_back(static_cast<float>(pw.x()),
                            static_cast<float>(pw.y()),
                            static_cast<float>(pw.z()));

        if (query_kp_idx < static_cast<int>(query.keypoints.size())) {
            pts_2d.push_back(query.keypoints[query_kp_idx].pt);
        }
    }

    int n_corr = std::min(static_cast<int>(pts_3d.size()), static_cast<int>(pts_2d.size()));
    if (n_corr < config_.min_pnp_inliers) return result;
    pts_3d.resize(n_corr);
    pts_2d.resize(n_corr);

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat rvec, tvec;
    cv::Mat inlier_mask;

    bool pnp_ok = cv::solvePnPRansac(pts_3d, pts_2d, K, cv::noArray(),
                                       rvec, tvec, false,
                                       config_.ransac_iterations,
                                       static_cast<float>(config_.ransac_reproj_thresh),
                                       0.99, inlier_mask);

    if (!pnp_ok || inlier_mask.empty()) return result;

    int num_inliers = cv::countNonZero(inlier_mask);
    if (num_inliers < config_.min_pnp_inliers) return result;

    // PnP → T_cam_world, convert to body frame
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    Eigen::Matrix3d R_cw;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R_cw(r, c) = R_cv.at<double>(r, c);
    Eigen::Vector3d t_cw(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    Eigen::Matrix3d R_wc = R_cw.transpose();
    Eigen::Vector3d t_wc = -R_wc * t_cw;

    Eigen::Matrix3d R_ci = T_cam_imu.block<3,3>(0,0);
    Eigen::Vector3d t_ci = T_cam_imu.block<3,1>(0,3);
    Eigen::Matrix3d R_ic = R_ci.transpose();
    Eigen::Vector3d t_ic = -R_ic * t_ci;

    // T_wb = T_wc * T_ci: R_wb = R_wc * R_ci, t_wb = R_wc * t_ci + t_wc
    Eigen::Matrix3d R_wb_query = R_wc * R_ci;
    Eigen::Vector3d t_wb_query = R_wc * t_ci + t_wc;

    // Return absolute body pose; main loop computes the relative transform
    result.query_timestamp = query.timestamp;
    result.match_timestamp = candidate.timestamp;
    result.relative_position = t_wb_query;
    result.relative_rotation = Eigen::Quaterniond(R_wb_query).normalized();
    result.num_inliers = num_inliers;
    result.valid = true;

    std::cout << "[LoopDetector] Loop detected! query_ts=" << query.timestamp
              << " match_ts=" << candidate.timestamp
              << " inliers=" << num_inliers << "/" << n_corr << std::endl;

    return result;
}

} // namespace vio
