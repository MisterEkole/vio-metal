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
#include "visualizer.h" 

#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <thread>

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
    if (argc < 2) {
        std::cerr << "Usage: ./vio-cpu <path_to_euroc_sequence>\n";
        return -1;
    }

    std::string dataset_path = argv[1];

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();
    vio::Profiler profiler; // Create Profiler instance

    // Prepare OpenCV calibration matrices
    cv::Mat K_left = (cv::Mat_<double>(3,3) << calib.intrinsics_left[0], 0, calib.intrinsics_left[2], 0, calib.intrinsics_left[1], calib.intrinsics_left[3], 0, 0, 1);
    cv::Mat dist_left = (cv::Mat_<double>(4,1) << calib.distortion_left[0], calib.distortion_left[1], calib.distortion_left[2], calib.distortion_left[3]);
    cv::Mat K_right = (cv::Mat_<double>(3,3) << calib.intrinsics_right[0], 0, calib.intrinsics_right[2], 0, calib.intrinsics_right[1], calib.intrinsics_right[3], 0, 0, 1);
    cv::Mat dist_right = (cv::Mat_<double>(4,1) << calib.distortion_right[0], calib.distortion_right[1], calib.distortion_right[2], calib.distortion_right[3]);

    // --- Stereo Rectification ---
    Eigen::Matrix3d R_rl_eigen = calib.T_cam1_cam0.block<3,3>(0,0);
    Eigen::Vector3d t_rl_eigen = calib.T_cam1_cam0.block<3,1>(0,3);
    cv::Mat R_rl_cv(3, 3, CV_64F), t_rl_cv(3, 1, CV_64F);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) R_rl_cv.at<double>(r,c) = R_rl_eigen(r,c);
        t_rl_cv.at<double>(r,0) = t_rl_eigen(r);
    }

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_left, dist_left, K_right, dist_right,
                      cv::Size(752, 480), R_rl_cv, t_rl_cv,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, cv::Size(752, 480));

    cv::Mat map_x_left, map_y_left, map_x_right, map_y_right;
    cv::initUndistortRectifyMap(K_left, dist_left, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
    cv::initUndistortRectifyMap(K_right, dist_right, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);

    // Update calibration for rectified frames
    double fx_rect = P1.at<double>(0,0);
    calib.intrinsics_left  = Eigen::Vector4d(fx_rect, P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2));
    double baseline_rect = -P2.at<double>(0,3) / fx_rect;
    calib.T_cam1_cam0 = Eigen::Matrix4d::Identity();
    calib.T_cam1_cam0(0,3) = -baseline_rect;

    // --- Pipeline Setup ---
    vio::FeatureDetector detector(vio::FeatureDetector::Config{});
    vio::StereoMatcher stereo_matcher(vio::StereoMatcher::Config{});
    vio::TemporalTracker tracker(vio::TemporalTracker::Config{});
    vio::FeatureManager feature_manager;
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});
    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::ImuPreintegrator preintegrator(vio::ImuNoiseParams{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    vio::VioVisualizer visualizer;
    auto ground_truth = loader.loadGroundTruth();

    std::thread vio_thread([&]() {
        bool pipeline_initialized = false;
        cv::Mat prev_left_undistorted;
        uint64_t frame_count = 0;
        uint64_t last_imu_ts = 0;
        int frames_since_keyframe = 0;

        while (loader.hasNext() && visualizer.isRunning()) {
            if (loader.nextIsImu()) {
                profiler.startStage("preintegration");
                vio::ImuSample imu = loader.getNextImuSample();
                if (pipeline_initialized) {
                    double dt = (last_imu_ts == 0) ? 0.005 : (imu.timestamp_ns - last_imu_ts) * 1e-9;
                    preintegrator.integrate(imu.gyro, imu.accel, dt);
                }
                last_imu_ts = imu.timestamp_ns;
                profiler.endStage("preintegration");
                continue;
            }

            vio::StereoFrame frame = loader.getNextStereoFrame();
            profiler.beginFrame(frame_count, frame.timestamp_ns);

            // --- STAGE: UNDISTORT ---
            profiler.startStage("undistort");
            cv::Mat left_undist, right_undist;
            cv::remap(frame.left,  left_undist,  map_x_left,  map_y_left,  cv::INTER_LINEAR);
            cv::remap(frame.right, right_undist, map_x_right, map_y_right, cv::INTER_LINEAR);
            profiler.endStage("undistort");

            Eigen::Vector3d current_est_pos = Eigen::Vector3d::Zero();
            bool pose_valid = false;

            if (!pipeline_initialized) {
                auto init_state = getInitialState(ground_truth, frame.timestamp_ns);
                preintegrator.reset(init_state.bias_gyro, init_state.bias_accel);
                optimizer.initialize(init_state);

                // --- INITIAL DETECTION ---
                profiler.startStage("detect");
                auto det = detector.detect(left_undist);
                auto det_right = detector.detect(right_undist);
                profiler.endStage("detect");

                profiler.startStage("stereo_match");
                auto stereo_matches = stereo_matcher.match(det.keypoints, det.descriptors, det_right.keypoints, det_right.descriptors, calib);
                profiler.endStage("stereo_match");
                
                feature_manager.addNewFeatures(frame.timestamp_ns, det.keypoints, det.descriptors, stereo_matches);
                pipeline_initialized = true;
                current_est_pos = init_state.position;
                pose_valid = true;
            } else {
                // --- STAGE: TEMPORAL TRACK ---
                profiler.startStage("temporal_track");
                auto prev_points = feature_manager.getCurrentPoints();
                auto track_result = tracker.track(prev_left_undistorted, left_undist, prev_points);
                feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), track_result.tracked_points, track_result.status);
                profiler.endStage("temporal_track");

                double parallax = vio::TemporalTracker::averageParallax(prev_points, track_result.tracked_points, track_result.status, calib.intrinsics_left[0]);
                frames_since_keyframe++;

                if (kf_policy.shouldInsertKeyframe(track_result.num_tracked, parallax, frames_since_keyframe)) {
                    // KEYFRAME PROCESSING
                    profiler.startStage("detect");
                    auto det_right = detector.detect(right_undist);
                    profiler.endStage("detect");

                    profiler.startStage("describe"); // Describe tracked points for stereo
                    std::vector<cv::KeyPoint> tracked_kpts;
                    for (const auto& pt : feature_manager.getCurrentPoints()) {
                        tracked_kpts.emplace_back(pt, 31.0f);
                    }
                    cv::Mat tracked_desc;
                    auto orb_tmp = cv::ORB::create();
                    orb_tmp->compute(left_undist, tracked_kpts, tracked_desc);
                    profiler.endStage("describe");

                    profiler.startStage("stereo_match");
                    auto tracked_stereo = stereo_matcher.match(tracked_kpts, tracked_desc, det_right.keypoints, det_right.descriptors, calib);
                    feature_manager.updateStereoForTracked(frame.timestamp_ns, feature_manager.getCurrentIds(), tracked_stereo);
                    profiler.endStage("stereo_match");

                    // ADD NEW FEATURES
                    profiler.startStage("detect");
                    auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 15);
                    profiler.endStage("detect");
                    
                    profiler.startStage("stereo_match");
                    auto new_stereo = stereo_matcher.match(new_det.keypoints, new_det.descriptors, det_right.keypoints, det_right.descriptors, calib);
                    feature_manager.addNewFeatures(frame.timestamp_ns, new_det.keypoints, new_det.descriptors, new_stereo);
                    profiler.endStage("stereo_match");

                    // --- STAGE: OPTIMIZE ---
                    profiler.startStage("optimize");
                    optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), feature_manager.getObservationsForFrame(frame.timestamp_ns));
                    optimizer.optimize();
                    profiler.endStage("optimize");
                    
                    auto latest_state = optimizer.latestState();
                    preintegrator.reset(latest_state.bias_gyro, latest_state.bias_accel);
                    frames_since_keyframe = 0;
                    current_est_pos = latest_state.position;
                    pose_valid = true;
                } else {
                    current_est_pos = optimizer.latestState().position; // Prediction logic simplified for visibility
                    pose_valid = true;
                }
            }

            prev_left_undistorted = left_undist.clone();
            if (pose_valid) {
                visualizer.addEstimate(current_est_pos);
                visualizer.addGroundTruth(getInitialState(ground_truth, frame.timestamp_ns).position);
            }

            profiler.endFrame();
            frame_count++;
        }
        profiler.printSummary(); // Output CPU metrics to console
        visualizer.stop();
    });

    visualizer.run();
    if (vio_thread.joinable()) vio_thread.join();

    return 0;
}