#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "dataset/EurocLoader.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./test_euroc_loader <path_to_euroc_sequence>\n";
        return -1;
    }

    std::string dataset_path = argv[1];
    std::cout << "Testing EuRoC Loader on dataset: " << dataset_path << std::endl;

    try {
        vio::EurocLoader loader(dataset_path);

        vio::StereoCalibration calib = loader.getCalibration();
        std::cout << "--- Calibration Loaded ---\n";
        std::cout << "Left Camera fx: " << calib.intrinsics_left[0] << std::endl;
        std::cout << "IMU Gyro Noise: " << calib.gyro_noise_density << std::endl;
        std::cout << "--------------------------\n";

        int imu_count = 0;
        int img_count = 0;

        while (loader.hasNext()) {
            if (loader.nextIsImu()) {
                vio::ImuSample imu = loader.getNextImuSample();
                imu_count++;
                if (imu_count % 1000 == 0) {
                    std::cout << "[IMU] Timestamp: " << imu.timestamp_ns 
                              << " | Accel X: " << imu.accel.x() << std::endl;
                }
            } 
            else if (loader.nextIsImage()) {
                vio::StereoFrame frame = loader.getNextStereoFrame();
                img_count++;
                
                cv::Mat stereo_display;
                cv::hconcat(frame.left, frame.right, stereo_display);
                cv::imshow("EuRoC Stereo Test", stereo_display);
                
                char key = (char)cv::waitKey(1);
                if (key == 27 || key == 'q') break;
            }
        }

        std::cout << "\nTest complete: " << imu_count << " IMU samples, " 
                  << img_count << " stereo frames." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}