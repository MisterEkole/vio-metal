#include "TrajectoryWriter.h"
#include <iomanip>
#include <iostream>

namespace vio {

TrajectoryWriter::TrajectoryWriter(const std::string& output_path)
    : file_(output_path)
{
    if (!file_.is_open()) {
        std::cerr << "[TrajectoryWriter] Failed to open: " << output_path << "\n";
    }
    file_ << std::fixed << std::setprecision(9);
}

TrajectoryWriter::~TrajectoryWriter() {
    if (file_.is_open()) file_.close();
}

void TrajectoryWriter::writePose(double timestamp_sec,
                                  const Eigen::Vector3d& position,
                                  const Eigen::Quaterniond& orientation)
{
    file_ << timestamp_sec << " "
          << position.x() << " "
          << position.y() << " "
          << position.z() << " "
          << orientation.x() << " "
          << orientation.y() << " "
          << orientation.z() << " "
          << orientation.w() << "\n";
}

void TrajectoryWriter::writePose(uint64_t timestamp_ns,
                                  const Eigen::Vector3d& position,
                                  const Eigen::Quaterniond& orientation)
{
    writePose(static_cast<double>(timestamp_ns) * 1e-9, position, orientation);
}

void TrajectoryWriter::flush() {
    file_.flush();
}

}
