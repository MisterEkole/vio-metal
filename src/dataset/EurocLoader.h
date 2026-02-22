#pragma once

#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace vio {

// Represents a synchronized stereo image pair
struct StereoFrame {
    uint64_t timestamp_ns;
    cv::Mat left;           // 752x480 grayscale
    cv::Mat right;          // 752x480 grayscale
};

// Represents a single IMU measurement
struct ImuSample {
    uint64_t timestamp_ns;
    Eigen::Vector3d gyro;   // rad/s
    Eigen::Vector3d accel;  // m/s^2
};

// Represents a ground truth pose
struct PoseStamped {
    uint64_t timestamp_ns;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bg; // Gyro bias
    Eigen::Vector3d ba; // Accel bias
};

// Calibration data loaded from sensor.yaml files
struct StereoCalibration {
    // Camera intrinsics [fx, fy, cx, cy]
    Eigen::Vector4d intrinsics_left;
    Eigen::Vector4d intrinsics_right;
    
    // Distortion [k1, k2, p1, p2]
    Eigen::Vector4d distortion_left;
    Eigen::Vector4d distortion_right;
    
    // Extrinsics
    Eigen::Matrix4d T_cam1_cam0; // Right cam relative to Left cam
    Eigen::Matrix4d T_cam0_imu;  // Left cam relative to IMU
    
    // IMU Noise parameters
    double gyro_noise_density;
    double gyro_random_walk;
    double accel_noise_density;
    double accel_random_walk;
};

class EurocLoader {
public:
    explicit EurocLoader(const std::string& dataset_path);
    ~EurocLoader() = default;

    // Temporal iteration interface
    bool hasNext() const;
    bool nextIsImage() const;
    bool nextIsImu() const;

    // Data retrieval
    StereoFrame getNextStereoFrame();
    ImuSample getNextImuSample();

    // Setup and evaluation
    std::vector<PoseStamped> loadGroundTruth() const;
    StereoCalibration getCalibration() const;

private:
    std::string dataset_path_;

    // Internal data structures to hold file paths and raw CSV data
    struct ImageRecord {
        uint64_t timestamp_ns;
        std::string filename_left;
        std::string filename_right;
    };

    std::vector<ImageRecord> image_records_;
    std::vector<ImuSample> imu_records_;

    // Iteration state
    size_t current_image_idx_ = 0;
    size_t current_imu_idx_ = 0;

    // Helper functions
    void loadImuData();
    void loadImageData();
    void skipCsvHeader(std::ifstream& file) const;
};

} // namespace vio