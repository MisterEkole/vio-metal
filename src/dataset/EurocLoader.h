#pragma once

#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace vio {

struct StereoFrame {
    uint64_t timestamp_ns;
    cv::Mat left;
    cv::Mat right;
};

struct ImuSample {
    uint64_t timestamp_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
};

struct PoseStamped {
    uint64_t timestamp_ns;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

struct StereoCalibration {
    Eigen::Vector4d intrinsics_left;   // [fx, fy, cx, cy]
    Eigen::Vector4d intrinsics_right;
    Eigen::Vector4d distortion_left;   // [k1, k2, p1, p2]
    Eigen::Vector4d distortion_right;
    Eigen::Matrix4d T_cam1_cam0;       // right cam relative to left
    Eigen::Matrix4d T_cam0_imu;        // left cam relative to IMU
    double gyro_noise_density;
    double gyro_random_walk;
    double accel_noise_density;
    double accel_random_walk;
};

class EurocLoader {
public:
    explicit EurocLoader(const std::string& dataset_path);
    ~EurocLoader() = default;

    bool hasNext() const;
    bool nextIsImage() const;
    bool nextIsImu() const;

    StereoFrame getNextStereoFrame();
    ImuSample getNextImuSample();

    std::vector<PoseStamped> loadGroundTruth() const;
    StereoCalibration getCalibration() const;

private:
    std::string dataset_path_;

    struct ImageRecord {
        uint64_t timestamp_ns;
        std::string filename_left;
        std::string filename_right;
    };

    std::vector<ImageRecord> image_records_;
    std::vector<ImuSample> imu_records_;

    size_t current_image_idx_ = 0;
    size_t current_imu_idx_ = 0;

    void loadImuData();
    void loadImageData();
    void skipCsvHeader(std::ifstream& file) const;
};

} // namespace vio