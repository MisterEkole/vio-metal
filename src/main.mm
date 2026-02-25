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
#include <thread>       // <-- THREADING

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

    // Build CPU rectified remap tables
    cv::Mat map_x_left, map_y_left, map_x_right, map_y_right;
    cv::initUndistortRectifyMap(K_left, dist_left, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
    cv::initUndistortRectifyMap(K_right, dist_right, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);

    // Update calibration to use rectified intrinsics
    double fx_rect = P1.at<double>(0,0);
    double fy_rect = P1.at<double>(1,1);
    double cx_rect = P1.at<double>(0,2);
    double cy_rect = P1.at<double>(1,2);
    calib.intrinsics_left  = Eigen::Vector4d(fx_rect, fy_rect, cx_rect, cy_rect);
    calib.intrinsics_right = Eigen::Vector4d(P2.at<double>(0,0), P2.at<double>(1,1),
                                              P2.at<double>(0,2), P2.at<double>(1,2));

    Eigen::Matrix3d R1_eigen;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R1_eigen(r,c) = R1.at<double>(r,c);
    Eigen::Matrix4d R1_4x4 = Eigen::Matrix4d::Identity();
    R1_4x4.block<3,3>(0,0) = R1_eigen;
    calib.T_cam0_imu = R1_4x4 * calib.T_cam0_imu;

    double baseline_rect = -P2.at<double>(0,3) / fx_rect;
    Eigen::Matrix4d T_cam1_cam0_rect = Eigen::Matrix4d::Identity();
    T_cam1_cam0_rect(0,3) = -baseline_rect; 
    calib.T_cam1_cam0 = T_cam1_cam0_rect;

    std::cerr << "[CPU Rectification] Initialized with fx=" << fx_rect << " baseline=" << baseline_rect << "\n";

    // --- Pipeline Setup ---
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

    vio::ImuPreintegrator preintegrator(imu_noise, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // =====================================================================
    // VIZUALIZER SETUP
    // =====================================================================
    vio::VioVisualizer visualizer;

    // =====================================================================
    // VIO BACKGROUND THREAD
    // =====================================================================
    std::thread vio_thread([&]() {
        bool pipeline_initialized = false;
        cv::Mat prev_left_undistorted;
        uint64_t frame_count = 0;
        uint64_t last_imu_ts = 0;
        int frames_since_keyframe = 0;

        while (loader.hasNext() && visualizer.isRunning()) {
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

            // --- CPU REMAPPING ---
            cv::Mat left_undist, right_undist;
            cv::remap(frame.left,  left_undist,  map_x_left,  map_y_left,  cv::INTER_LINEAR);
            cv::remap(frame.right, right_undist, map_x_right, map_y_right, cv::INTER_LINEAR);

            Eigen::Vector3d current_est_pos = Eigen::Vector3d::Zero();
            bool pose_valid = false;

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

                auto init_obs = feature_manager.getObservationsForFrame(frame.timestamp_ns);
                optimizer.addObservations(frame.timestamp_ns, init_obs);

                traj_writer.writePose(frame.timestamp_ns, init_state.position, init_state.rotation);
                prev_left_undistorted = left_undist.clone();
                pipeline_initialized = true;

                current_est_pos = init_state.position;
                pose_valid = true;
                
            } else {
                auto prev_points = feature_manager.getCurrentPoints();
                auto track_result = tracker.track(prev_left_undistorted, left_undist, prev_points);
                feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), track_result.tracked_points, track_result.status);

                double parallax = vio::TemporalTracker::averageParallax(prev_points, track_result.tracked_points, track_result.status, calib.intrinsics_left[0]);
                frames_since_keyframe++;

                if (kf_policy.shouldInsertKeyframe(track_result.num_tracked, parallax, frames_since_keyframe)) {
                    auto tracked_pts = feature_manager.getCurrentPoints();
                    auto tracked_ids = feature_manager.getCurrentIds();
                    
                    std::vector<cv::KeyPoint> tracked_kpts;
                    for (size_t i = 0; i < tracked_pts.size(); i++) {
                        cv::KeyPoint kp(tracked_pts[i], 31.0f);
                        kp.class_id = static_cast<int>(i);
                        tracked_kpts.push_back(kp);
                    }
                    
                    cv::Mat tracked_desc;
                    auto orb_tmp = cv::ORB::create();
                    orb_tmp->compute(left_undist, tracked_kpts, tracked_desc);
                    
                    auto det_right = detector.detect(right_undist);
                    auto tracked_stereo = stereo_matcher.match(tracked_kpts, tracked_desc, det_right.keypoints, det_right.descriptors, calib);
                    
                    for (auto& m : tracked_stereo) {
                        m.left_idx = tracked_kpts[m.left_idx].class_id;
                    }
                    
                    feature_manager.updateStereoForTracked(frame.timestamp_ns, tracked_ids, tracked_stereo);
                    
                    auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 15);
                    auto new_stereo = stereo_matcher.match(new_det.keypoints, new_det.descriptors, det_right.keypoints, det_right.descriptors, calib);
                    feature_manager.addNewFeatures(frame.timestamp_ns, new_det.keypoints, new_det.descriptors, new_stereo);

                    auto current_state = optimizer.latestState();
                    auto world_lms = feature_manager.getInitializedLandmarksWorld(current_state.position, current_state.rotation, calib.T_cam0_imu);
                    optimizer.setLandmarks(world_lms);

                    auto frame_obs = feature_manager.getObservationsForFrame(frame.timestamp_ns);
                    optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), frame_obs);

                    optimizer.optimize();
                    
                    auto latest_state = optimizer.latestState();
                    traj_writer.writePose(frame.timestamp_ns, latest_state.position, latest_state.rotation);
                    preintegrator.reset(latest_state.bias_gyro, latest_state.bias_accel);
                    frames_since_keyframe = 0;

                    current_est_pos = latest_state.position;
                    pose_valid = true;
                } else {
                    auto prev_state = optimizer.latestState();
                    auto preint_result = preintegrator.getResult();
                    if (preint_result.dt > 1e-7) {
                        Eigen::Matrix3d R_prev = prev_state.rotation.toRotationMatrix();
                        Eigen::Vector3d g = vio::gravity();
                        double dt = preint_result.dt;
                        Eigen::Vector3d pred_pos = prev_state.position + prev_state.velocity * dt + 0.5 * g * dt * dt + R_prev * preint_result.delta_p;
                        Eigen::Quaterniond pred_rot = prev_state.rotation * preint_result.delta_R;
                        pred_rot.normalize();
                        traj_writer.writePose(frame.timestamp_ns, pred_pos, pred_rot);

                        current_est_pos = pred_pos;
                        pose_valid = true;
                    }
                }

                prev_left_undistorted = left_undist.clone();
            }

            // =====================================================================
            // FEED VIZUALIZER
            // =====================================================================
            if (pose_valid) {
                auto current_gt = getInitialState(ground_truth, frame.timestamp_ns);
                
                // Note: aligned_gt_pos subtraction has been intentionally removed 
                // so raw positions are pushed to the visualizer.
                visualizer.addEstimate(current_est_pos);
                visualizer.addGroundTruth(current_gt.position);
            }

            frame_count++;
            profiler.endFrame();
        }
        visualizer.stop(); // Safe shutdown
    });

    // =====================================================================
    // RUN VIZUALIZER (MUST BE ON MAIN THREAD)
    // =====================================================================
    visualizer.run();

    if (vio_thread.joinable()) {
        vio_thread.join();
    }

    return 0;
}