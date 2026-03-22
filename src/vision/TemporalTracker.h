#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vio {

class TemporalTracker {
public:
    struct Config {
        cv::Size win_size = cv::Size(11, 11);
        int max_level = 3;
        int max_iterations = 15;
        double epsilon = 0.01;
        double max_error = 50.0;         // Pixels — reject if track error exceeds this
        double min_eigen_threshold = 1e-4;
        bool use_forward_backward = false; // FB check filters tracking outliers
    };
    TemporalTracker();

    explicit TemporalTracker(const Config& config);

    struct TrackResult {
        std::vector<cv::Point2f> tracked_points;
        std::vector<bool> status;   // true = successfully tracked
        int num_tracked = 0;
        double track_ms = 0.0;
    };

    TrackResult track(const cv::Mat& prev_image,
                      const cv::Mat& curr_image,
                      const std::vector<cv::Point2f>& prev_points);

    static double averageParallax(const std::vector<cv::Point2f>& prev,
                                  const std::vector<cv::Point2f>& curr,
                                  const std::vector<bool>& status,
                                  double focal_length = 460.0);

private:
    Config config_;
};

} 
