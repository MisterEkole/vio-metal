#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace vio {

class TrajectoryWriter {
public:
    explicit TrajectoryWriter(const std::string& output_path);
    ~TrajectoryWriter();

    // Write a single pose in TUM format: timestamp tx ty tz qx qy qz qw
    void writePose(double timestamp_sec,
                   const Eigen::Vector3d& position,
                   const Eigen::Quaterniond& orientation);

    void writePose(uint64_t timestamp_ns,
                   const Eigen::Vector3d& position,
                   const Eigen::Quaterniond& orientation);

    void flush();

private:
    std::ofstream file_;
};

} 
