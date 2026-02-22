#include "FeatureDetector.h"
#include <chrono>
#include <algorithm>

namespace vio {


FeatureDetector::FeatureDetector() 
    : FeatureDetector(Config{}) 
{}


FeatureDetector::FeatureDetector(const Config& config)
    : config_(config)
{
    orb_ = cv::ORB::create(
        config_.max_features,
        config_.orb_scale_factor,
        config_.orb_nlevels,
        31,  // edgeThreshold
        0,   // firstLevel
        2,   // WTA_K
        cv::ORB::HARRIS_SCORE,
        31,  // patchSize
        config_.fast_threshold
    );
}

FeatureDetector::DetectionResult FeatureDetector::detect(const cv::Mat& image) {
    DetectionResult result;
    if (image.empty()) return result; // Safety check

    auto t0 = std::chrono::high_resolution_clock::now();

    // Grid-based FAST detection to ensure spatial distribution
    std::vector<cv::KeyPoint> all_kpts;
    int cell_h = image.rows / config_.grid_rows;
    int cell_w = image.cols / config_.grid_cols;
    int per_cell = (config_.max_features * 2) / (config_.grid_rows * config_.grid_cols);

    for (int r = 0; r < config_.grid_rows; r++) {
        for (int c = 0; c < config_.grid_cols; c++) {
            int x0 = c * cell_w;
            int y0 = r * cell_h;
            int w = (c == config_.grid_cols - 1) ? (image.cols - x0) : cell_w;
            int h = (r == config_.grid_rows - 1) ? (image.rows - y0) : cell_h;

            cv::Rect roi(x0, y0, w, h);
            cv::Mat cell = image(roi);

            std::vector<cv::KeyPoint> cell_kpts;
            cv::FAST(cell, cell_kpts, config_.fast_threshold, true);

            std::sort(cell_kpts.begin(), cell_kpts.end(),
                      [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                          return a.response > b.response;
                      });

            int keep = std::min((int)cell_kpts.size(), per_cell);
            for (int i = 0; i < keep; i++) {
                cell_kpts[i].pt.x += x0;
                cell_kpts[i].pt.y += y0;
                all_kpts.push_back(cell_kpts[i]);
            }
        }
    }

    distributeFeatures(all_kpts, image.cols, image.rows);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.keypoints = all_kpts;
    if (!result.keypoints.empty()) {
        orb_->compute(image, result.keypoints, result.descriptors);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    result.describe_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    return result;
}

FeatureDetector::DetectionResult FeatureDetector::detect(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& existing_points,
    int mask_radius)
{
    if (existing_points.empty()) {
        return detect(image);
    }

    cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;
    for (const auto& pt : existing_points) {
        cv::circle(mask, pt, mask_radius, cv::Scalar(0), -1);
    }

    DetectionResult result = detect(image);

    std::vector<cv::KeyPoint> filtered_kpts;
    cv::Mat filtered_desc;
    
    // Safety check for descriptors before accessing rows
    if (result.descriptors.empty()) return result;

    for (size_t i = 0; i < result.keypoints.size(); i++) {
        const auto& kp = result.keypoints[i];
        int px = static_cast<int>(kp.pt.x);
        int py = static_cast<int>(kp.pt.y);
        if (px >= 0 && px < mask.cols && py >= 0 && py < mask.rows &&
            mask.at<uint8_t>(py, px) > 0) {
            filtered_kpts.push_back(kp);
            filtered_desc.push_back(result.descriptors.row(static_cast<int>(i)));
        }
    }

    result.keypoints = filtered_kpts;
    result.descriptors = filtered_desc;
    return result;
}

void FeatureDetector::distributeFeatures(std::vector<cv::KeyPoint>& kpts,
                                          int image_width, int image_height) const
{
    if (kpts.empty()) return;

    std::sort(kpts.begin(), kpts.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    std::vector<cv::KeyPoint> result;
    std::vector<bool> suppressed(kpts.size(), false);
    double min_dist_sq = config_.min_distance * config_.min_distance;

    for (size_t i = 0; i < kpts.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(kpts[i]);

        if ((int)result.size() >= config_.max_features) break;

        for (size_t j = i + 1; j < kpts.size(); j++) {
            if (suppressed[j]) continue;
            double dx = kpts[i].pt.x - kpts[j].pt.x;
            double dy = kpts[i].pt.y - kpts[j].pt.y;
            if (dx * dx + dy * dy < min_dist_sq) {
                suppressed[j] = true;
            }
        }
    }
    kpts = result;
}

} 