#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace vio {

class FeatureDetector {
public:
    struct Config {
        int max_features = 500;
        int fast_threshold = 20;
        int orb_nlevels = 4;
        float orb_scale_factor = 1.2f;
        int grid_rows = 4;
        int grid_cols = 5;
        int min_distance = 15;
    };
    FeatureDetector();
    explicit FeatureDetector(const Config& config);

    struct DetectionResult {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;           // ORB: CV_8U, 32 bytes per row
        double detect_ms = 0.0;
        double describe_ms = 0.0;
    };

    // Detect keypoints and extract ORB descriptors
    DetectionResult detect(const cv::Mat& image);

    // Detect in masked region (avoid re-detecting already tracked points)
    DetectionResult detect(const cv::Mat& image,
                           const std::vector<cv::Point2f>& existing_points,
                           int mask_radius = 15);

private:
    Config config_;
    cv::Ptr<cv::ORB> orb_;

    // Grid-based feature distribution
    void distributeFeatures(std::vector<cv::KeyPoint>& kpts,
                            int image_width, int image_height) const;
};

}
