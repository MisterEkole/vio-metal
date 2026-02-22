#include "dataset/EurocLoader.h"
#include "dataset/TrajectoryWriter.h"
#include "core/Types.h"
#include "core/Profiler.h"
#include "core/KeyframePolicy.h"
#include "vision/FeatureDetector.h"
#include "vision/StereoMatcher.h"
#include "vision/TemporalTracker.h"
#include "vision/FeatureManager.h"
#include "imu/ImuPreintegrator.h"
#include "imu/ImuTypes.h"
#include "optimization/VioOptimizer.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>

#import <Foundation/Foundation.h>
#include "metal/MetalContext.h" 
#include "metal/MetalUndistort.h"

namespace {
vio::KeyframeState getInitialState(const std::vector<vio::PoseStamped>& gt, uint64_t target_ts) {
    vio::KeyframeState state;
    state.timestamp = target_ts;
    state.position.setZero();
    state.rotation.setIdentity();
    state.velocity.setZero();
    state.bias_gyro.setZero();
    state.bias_accel.setZero();

    uint64_t best_diff = UINT64_MAX;
    for (const auto& p : gt) {
        uint64_t diff = (p.timestamp_ns > target_ts) ?
            (p.timestamp_ns - target_ts) : (target_ts - p.timestamp_ns);
        if (diff < best_diff) {
            best_diff = diff;
            state.position = p.position;
            state.rotation = p.orientation;
            state.velocity = p.velocity;
            state.bias_gyro = p.bg;
            state.bias_accel = p.ba;
        }
    }
    return state;
}
} 

int main(int argc, char** argv) {
    @autoreleasepool {

    if (argc < 2) {
        std::cerr << "Usage: ./vio-metal <path_to_euroc_sequence> [metallib_path]\n";
        return -1;
    }

    std::string dataset_path = argv[1];
    std::string metallib_path = (argc >= 3) ? argv[2] : "undistort.metallib";

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();

    cv::Mat K_left = (cv::Mat_<double>(3,3) << calib.intrinsics_left[0], 0, calib.intrinsics_left[2], 0, calib.intrinsics_left[1], calib.intrinsics_left[3], 0, 0, 1);
    cv::Mat dist_left = (cv::Mat_<double>(4,1) << calib.distortion_left[0], calib.distortion_left[1], calib.distortion_left[2], calib.distortion_left[3]);
    cv::Mat K_right = (cv::Mat_<double>(3,3) << calib.intrinsics_right[0], 0, calib.intrinsics_right[2], 0, calib.intrinsics_right[1], calib.intrinsics_right[3], 0, 0, 1);
    cv::Mat dist_right = (cv::Mat_<double>(4,1) << calib.distortion_right[0], calib.distortion_right[1], calib.distortion_right[2], calib.distortion_right[3]);

    cv::Mat map_x_left, map_y_left, map_x_right, map_y_right;
    cv::initUndistortRectifyMap(K_left, dist_left, cv::Mat(), K_left, cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
    cv::initUndistortRectifyMap(K_right, dist_right, cv::Mat(), K_right, cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);

    vio::MetalContext* metal_context = nullptr; 
    vio::MetalUndistort* metal_undistort_left = nullptr;
    vio::MetalUndistort* metal_undistort_right = nullptr;
    bool use_metal = false;

    if (vio::MetalContext::isAvailable()) {
        metal_context = new vio::MetalContext();
        try {
            metal_undistort_left = new vio::MetalUndistort(metal_context, map_x_left, map_y_left, 752, 480, metallib_path);
            metal_undistort_right = new vio::MetalUndistort(metal_context, map_x_right, map_y_right, 752, 480, metallib_path);
            use_metal = metal_undistort_left->isReady() && metal_undistort_right->isReady();
        } catch (const std::exception& e) {
            std::cerr << "[Metal] GPU Init Error: " << e.what() << "\n";
        }
    }

    vio::FeatureDetector detector(vio::FeatureDetector::Config{});
    vio::StereoMatcher stereo_matcher(vio::StereoMatcher::Config{});
    vio::TemporalTracker tracker(vio::TemporalTracker::Config{});
    vio::FeatureManager feature_manager;
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});

    vio::ImuNoiseParams imu_noise;
    imu_noise.gyro_noise_density = calib.gyro_noise_density;
    imu_noise.accel_noise_density = calib.accel_noise_density;
    imu_noise.gyro_random_walk = calib.gyro_random_walk;
    imu_noise.accel_random_walk = calib.accel_random_walk;

    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::Profiler profiler;
    vio::TrajectoryWriter traj_writer("results/trajectories/estimated.txt");
    auto ground_truth = loader.loadGroundTruth();

    bool pipeline_initialized = false;
    cv::Mat prev_left_undistorted;
    uint64_t frame_count = 0;
    uint64_t last_imu_ts = 0;
    int frames_since_keyframe = 0;

    vio::ImuPreintegrator preintegrator(imu_noise, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    while (loader.hasNext()) {
        if (loader.nextIsImu()) {
            vio::ImuSample imu = loader.getNextImuSample();
            if (pipeline_initialized) {
                double dt = (last_imu_ts == 0) ? 0.005 : (imu.timestamp_ns - last_imu_ts) * 1e-9;
                preintegrator.integrate(imu.gyro, imu.accel, dt);
            }
            last_imu_ts = imu.timestamp_ns;
            continue;
        }

        profiler.beginFrame(frame_count, 0);
        vio::StereoFrame frame = loader.getNextStereoFrame();

        cv::Mat left_undist, right_undist;
        if (use_metal) {
            left_undist  = metal_undistort_left->undistort(frame.left);
            right_undist = metal_undistort_right->undistort(frame.right);
        } else {
            cv::remap(frame.left,  left_undist,  map_x_left,  map_y_left,  cv::INTER_LINEAR);
            cv::remap(frame.right, right_undist, map_x_right, map_y_right, cv::INTER_LINEAR);
        }

        if (!pipeline_initialized) {
            auto init_state = getInitialState(ground_truth, frame.timestamp_ns);
            preintegrator.reset(init_state.bias_gyro, init_state.bias_accel);
            optimizer.initialize(init_state);

            auto det = detector.detect(left_undist);
            auto det_right = detector.detect(right_undist);
            auto stereo_matches = stereo_matcher.match(det.keypoints, det.descriptors, det_right.keypoints, det_right.descriptors, calib);
            
            feature_manager.addNewFeatures(frame.timestamp_ns, det.keypoints, det.descriptors, stereo_matches);
            
            auto world_lms = feature_manager.getInitializedLandmarksWorld(init_state.position, init_state.rotation, calib.T_cam0_imu);
            optimizer.setLandmarks(world_lms);

            prev_left_undistorted = left_undist.clone();
            pipeline_initialized = true;
            frame_count++;
            continue;
        }

        auto prev_points = feature_manager.getCurrentPoints();
        auto track_result = tracker.track(prev_left_undistorted, left_undist, prev_points);
        feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), track_result.tracked_points, track_result.status);

        double parallax = vio::TemporalTracker::averageParallax(prev_points, track_result.tracked_points, track_result.status, calib.intrinsics_left[0]);
        frames_since_keyframe++;

        if (kf_policy.shouldInsertKeyframe(track_result.num_tracked, parallax, frames_since_keyframe)) {
            auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 15);
            auto det_right = detector.detect(right_undist);
            auto stereo_matches = stereo_matcher.match(new_det.keypoints, new_det.descriptors, det_right.keypoints, det_right.descriptors, calib);
            feature_manager.addNewFeatures(frame.timestamp_ns, new_det.keypoints, new_det.descriptors, stereo_matches);

            auto current_state = optimizer.latestState();
            auto world_lms = feature_manager.getInitializedLandmarksWorld(current_state.position, current_state.rotation, calib.T_cam0_imu);
            optimizer.setLandmarks(world_lms);

            optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), feature_manager.getObservationsForFrame(frame.timestamp_ns));
            optimizer.optimize();
            
            auto latest_state = optimizer.latestState();
            traj_writer.writePose(frame.timestamp_ns, latest_state.position, latest_state.rotation);
            preintegrator.reset(latest_state.bias_gyro, latest_state.bias_accel);
            frames_since_keyframe = 0;
        }

        prev_left_undistorted = left_undist.clone();
        frame_count++;
        profiler.endFrame();
    }

    delete metal_undistort_left;
    delete metal_undistort_right;
    delete metal_context; 

    }
    return 0;
}