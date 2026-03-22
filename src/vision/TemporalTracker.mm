#include "TemporalTracker.h"
#include <chrono>
#include <cmath>

namespace vio {
TemporalTracker::TemporalTracker():TemporalTracker(Config{}){}

TemporalTracker::TemporalTracker(const Config& config)
    : config_(config) {}

TemporalTracker::TrackResult TemporalTracker::track(
    const cv::Mat& prev_image,
    const cv::Mat& curr_image,
    const std::vector<cv::Point2f>& prev_points)
{
    TrackResult result;
    result.tracked_points.resize(prev_points.size());
    result.status.resize(prev_points.size(), false);
    result.num_tracked = 0;

    if (prev_points.empty()) return result;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Point2f> fwd_pts;
    std::vector<uchar> fwd_status;
    std::vector<float> fwd_err;

    cv::TermCriteria criteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        config_.max_iterations, config_.epsilon
    );

    cv::calcOpticalFlowPyrLK(
        prev_image, curr_image,
        prev_points, fwd_pts,
        fwd_status, fwd_err,
        config_.win_size, config_.max_level,
        criteria,
        0, config_.min_eigen_threshold
    );

    if (config_.use_forward_backward) {
        // Backward pass for FB consistency
        std::vector<cv::Point2f> bwd_pts;
        std::vector<uchar> bwd_status;
        std::vector<float> bwd_err;

        cv::calcOpticalFlowPyrLK(
            curr_image, prev_image,
            fwd_pts, bwd_pts,
            bwd_status, bwd_err,
            config_.win_size, config_.max_level,
            criteria,
            0, config_.min_eigen_threshold
        );

        for (size_t i = 0; i < prev_points.size(); i++) {
            if (!fwd_status[i] || !bwd_status[i]) continue;

            double fb_dx = prev_points[i].x - bwd_pts[i].x;
            double fb_dy = prev_points[i].y - bwd_pts[i].y;
            double fb_err = std::sqrt(fb_dx * fb_dx + fb_dy * fb_dy);
            if (fb_err > 1.0) continue;

            if (fwd_err[i] > config_.max_error) continue;

            if (fwd_pts[i].x < 0 || fwd_pts[i].x >= curr_image.cols ||
                fwd_pts[i].y < 0 || fwd_pts[i].y >= curr_image.rows) continue;

            result.tracked_points[i] = fwd_pts[i];
            result.status[i] = true;
            result.num_tracked++;
        }
    } else {
        for (size_t i = 0; i < prev_points.size(); i++) {
            if (!fwd_status[i]) continue;

            if (fwd_err[i] > config_.max_error) continue;

            if (fwd_pts[i].x < 0 || fwd_pts[i].x >= curr_image.cols ||
                fwd_pts[i].y < 0 || fwd_pts[i].y >= curr_image.rows) continue;

            result.tracked_points[i] = fwd_pts[i];
            result.status[i] = true;
            result.num_tracked++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.track_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

double TemporalTracker::averageParallax(
    const std::vector<cv::Point2f>& prev,
    const std::vector<cv::Point2f>& curr,
    const std::vector<bool>& status,
    double focal_length)
{
    double total = 0.0;
    int count = 0;
    for (size_t i = 0; i < prev.size() && i < curr.size(); i++) {
        if (i < status.size() && !status[i]) continue;
        double dx = curr[i].x - prev[i].x;
        double dy = curr[i].y - prev[i].y;
        double pixel_disp = std::sqrt(dx * dx + dy * dy);
        total += std::atan2(pixel_disp, focal_length) * 180.0 / M_PI;
        count++;
    }
    return (count > 0) ? total / count : 0.0;
}

}
