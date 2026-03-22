#include "EurocLoader.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <yaml-cpp/yaml.h> 

namespace vio {

EurocLoader::EurocLoader(const std::string& dataset_path) 
    : dataset_path_(dataset_path) {
    
    if (!std::filesystem::exists(dataset_path_)) {
        throw std::runtime_error("Dataset path does not exist: " + dataset_path_);
    }

    loadImuData();
    loadImageData();
}

void EurocLoader::skipCsvHeader(std::ifstream& file) const {
    std::string line;
    std::getline(file, line);
}

void EurocLoader::loadImuData() {
    std::string imu_csv_path = dataset_path_ + "/mav0/imu0/data.csv";
    std::ifstream file(imu_csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open IMU data.csv");
    }

    skipCsvHeader(file);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        ImuSample sample;

        std::getline(ss, token, ','); sample.timestamp_ns = std::stoull(token);
        std::getline(ss, token, ','); sample.gyro.x() = std::stod(token);
        std::getline(ss, token, ','); sample.gyro.y() = std::stod(token);
        std::getline(ss, token, ','); sample.gyro.z() = std::stod(token);
        std::getline(ss, token, ','); sample.accel.x() = std::stod(token);
        std::getline(ss, token, ','); sample.accel.y() = std::stod(token);
        std::getline(ss, token, ','); sample.accel.z() = std::stod(token);

        imu_records_.push_back(sample);
    }
}

void EurocLoader::loadImageData() {
    std::string cam0_csv_path = dataset_path_ + "/mav0/cam0/data.csv";
    std::string cam1_csv_path = dataset_path_ + "/mav0/cam1/data.csv";

    std::ifstream file0(cam0_csv_path);
    std::ifstream file1(cam1_csv_path);

    if (!file0.is_open() || !file1.is_open()) {
        throw std::runtime_error("Failed to open camera data.csv files");
    }

    skipCsvHeader(file0);
    skipCsvHeader(file1);

    std::string line0, line1;
    while (std::getline(file0, line0) && std::getline(file1, line1)) {
        if (line0.empty() || line1.empty()) continue;

        std::stringstream ss0(line0);
        std::stringstream ss1(line1);
        std::string token0, token1;

        ImageRecord record;
        
        std::getline(ss0, token0, ','); record.timestamp_ns = std::stoull(token0);
        std::getline(ss0, token0, ','); record.filename_left = token0;

        std::getline(ss1, token1, ',');
        std::getline(ss1, token1, ','); record.filename_right = token1;

        if (!record.filename_left.empty() && record.filename_left.back() == '\r') record.filename_left.pop_back();
        if (!record.filename_right.empty() && record.filename_right.back() == '\r') record.filename_right.pop_back();

        image_records_.push_back(record);
    }
}

bool EurocLoader::hasNext() const {
    return current_image_idx_ < image_records_.size() || current_imu_idx_ < imu_records_.size();
}

bool EurocLoader::nextIsImage() const {
    if (current_image_idx_ >= image_records_.size()) return false;
    if (current_imu_idx_ >= imu_records_.size()) return true;

    return image_records_[current_image_idx_].timestamp_ns <= imu_records_[current_imu_idx_].timestamp_ns;
}

bool EurocLoader::nextIsImu() const {
    if (current_imu_idx_ >= imu_records_.size()) return false;
    if (current_image_idx_ >= image_records_.size()) return true;

    return imu_records_[current_imu_idx_].timestamp_ns < image_records_[current_image_idx_].timestamp_ns;
}

StereoFrame EurocLoader::getNextStereoFrame() {
    if (!hasNext() || !nextIsImage()) {
        throw std::runtime_error("Next event is not an image.");
    }

    const auto& record = image_records_[current_image_idx_++];
    
    StereoFrame frame;
    frame.timestamp_ns = record.timestamp_ns;
    
    std::string left_path = dataset_path_ + "/mav0/cam0/data/" + record.filename_left;
    std::string right_path = dataset_path_ + "/mav0/cam1/data/" + record.filename_right;

    frame.left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    frame.right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

    if (frame.left.empty() || frame.right.empty()) {
        throw std::runtime_error("Failed to load images at timestamp: " + std::to_string(frame.timestamp_ns));
    }

    return frame;
}

ImuSample EurocLoader::getNextImuSample() {
    if (!hasNext() || !nextIsImu()) {
        throw std::runtime_error("Next event is not an IMU sample.");
    }
    return imu_records_[current_imu_idx_++];
}

std::vector<PoseStamped> EurocLoader::loadGroundTruth() const {
    std::vector<PoseStamped> gt_trajectory;
    std::string gt_csv_path = dataset_path_ + "/mav0/state_groundtruth_estimate0/data.csv";
    std::ifstream file(gt_csv_path);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Ground truth not found at " << gt_csv_path << "\n";
        return gt_trajectory;
    }

    skipCsvHeader(file);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string token;
        PoseStamped pose;

        std::getline(ss, token, ','); pose.timestamp_ns = std::stoull(token);
        std::getline(ss, token, ','); pose.position.x() = std::stod(token);
        std::getline(ss, token, ','); pose.position.y() = std::stod(token);
        std::getline(ss, token, ','); pose.position.z() = std::stod(token);
        
        double qw, qx, qy, qz;
        std::getline(ss, token, ','); qw = std::stod(token);
        std::getline(ss, token, ','); qx = std::stod(token);
        std::getline(ss, token, ','); qy = std::stod(token);
        std::getline(ss, token, ','); qz = std::stod(token);
        pose.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
        
        std::getline(ss, token, ','); pose.velocity.x() = std::stod(token);
        std::getline(ss, token, ','); pose.velocity.y() = std::stod(token);
        std::getline(ss, token, ','); pose.velocity.z() = std::stod(token);

        std::getline(ss, token, ','); pose.bg.x() = std::stod(token);
        std::getline(ss, token, ','); pose.bg.y() = std::stod(token);
        std::getline(ss, token, ','); pose.bg.z() = std::stod(token);
        std::getline(ss, token, ','); pose.ba.x() = std::stod(token);
        std::getline(ss, token, ','); pose.ba.y() = std::stod(token);
        std::getline(ss, token, ','); pose.ba.z() = std::stod(token);

        gt_trajectory.push_back(pose);
    }
    return gt_trajectory;
}

StereoCalibration EurocLoader::getCalibration() const {
    StereoCalibration calib;
    
    try {
        YAML::Node cam0 = YAML::LoadFile(dataset_path_ + "/mav0/cam0/sensor.yaml");
        YAML::Node cam1 = YAML::LoadFile(dataset_path_ + "/mav0/cam1/sensor.yaml");
        YAML::Node imu = YAML::LoadFile(dataset_path_ + "/mav0/imu0/sensor.yaml");

        auto intr0 = cam0["intrinsics"].as<std::vector<double>>();
        calib.intrinsics_left = Eigen::Vector4d(intr0[0], intr0[1], intr0[2], intr0[3]);
        
        auto dist0 = cam0["distortion_coefficients"].as<std::vector<double>>();
        calib.distortion_left = Eigen::Vector4d(dist0[0], dist0[1], dist0[2], dist0[3]);

        auto intr1 = cam1["intrinsics"].as<std::vector<double>>();
        calib.intrinsics_right = Eigen::Vector4d(intr1[0], intr1[1], intr1[2], intr1[3]);
        
        auto dist1 = cam1["distortion_coefficients"].as<std::vector<double>>();
        calib.distortion_right = Eigen::Vector4d(dist1[0], dist1[1], dist1[2], dist1[3]);

        // T_BS: body (IMU) to sensor
        auto T_BS_cam0 = cam0["T_BS"]["data"].as<std::vector<double>>();
        auto T_BS_cam1 = cam1["T_BS"]["data"].as<std::vector<double>>();
        
        Eigen::Matrix4d T_imu_cam0 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_BS_cam0.data());
        Eigen::Matrix4d T_imu_cam1 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_BS_cam1.data());

        calib.T_cam0_imu = T_imu_cam0.inverse();
        calib.T_cam1_cam0 = T_imu_cam1.inverse() * T_imu_cam0;

        calib.gyro_noise_density = imu["gyroscope_noise_density"].as<double>();
        calib.gyro_random_walk = imu["gyroscope_random_walk"].as<double>();
        calib.accel_noise_density = imu["accelerometer_noise_density"].as<double>();
        calib.accel_random_walk = imu["accelerometer_random_walk"].as<double>();
        
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to parse calibration YAML: " << e.what() << "\n";
    }

    return calib;
}

} 