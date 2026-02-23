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

    // --- Stereo Rectification ---
    // Extract R and t from T_cam1_cam0 (transforms cam0 points into cam1 frame)
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

    // Build rectified remap tables (undistort + rectify in one step)
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

    // Update T_cam0_imu: rectification rotates the camera frame by R1
    // p_rect = R1 * p_cam = R1 * (T_cam0_imu * p_imu)
    // So T_cam0rect_imu = R1_4x4 * T_cam0_imu  (PRE-multiply)
    Eigen::Matrix3d R1_eigen;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R1_eigen(r,c) = R1.at<double>(r,c);
    Eigen::Matrix4d R1_4x4 = Eigen::Matrix4d::Identity();
    R1_4x4.block<3,3>(0,0) = R1_eigen;
    calib.T_cam0_imu = R1_4x4 * calib.T_cam0_imu;

    // Update T_cam1_cam0: after rectification, cameras are axis-aligned with pure x-baseline
    // Baseline from P2: P2[0][3] = -fx * baseline
    double baseline_rect = -P2.at<double>(0,3) / fx_rect;
    Eigen::Matrix4d T_cam1_cam0_rect = Eigen::Matrix4d::Identity();
    T_cam1_cam0_rect(0,3) = -baseline_rect;  // cam1 is shifted left of cam0 by baseline
    calib.T_cam1_cam0 = T_cam1_cam0_rect;

    std::cerr << "[Rectification] fx=" << fx_rect << " fy=" << fy_rect
              << " cx=" << cx_rect << " cy=" << cy_rect
              << " baseline=" << baseline_rect << "\n";

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

            // Anchor landmarks: register init frame observations so the fixed
            // first keyframe has visual residuals that constrain landmark positions
            auto init_obs = feature_manager.getObservationsForFrame(frame.timestamp_ns);
            optimizer.addObservations(frame.timestamp_ns, init_obs);

            traj_writer.writePose(frame.timestamp_ns, init_state.position, init_state.rotation);

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
            // --- Stereo-match TRACKED features (the key to getting landmarks) ---
            auto tracked_pts = feature_manager.getCurrentPoints();
            auto tracked_ids = feature_manager.getCurrentIds();
            
            // Create KeyPoints from tracked positions with index tracking
            // ORB::compute may remove keypoints near borders, so we store
            // the original index in class_id to maintain the mapping
            std::vector<cv::KeyPoint> tracked_kpts;
            for (size_t i = 0; i < tracked_pts.size(); i++) {
                cv::KeyPoint kp(tracked_pts[i], 31.0f);
                kp.class_id = static_cast<int>(i);  // maps back to tracked_ids[i]
                tracked_kpts.push_back(kp);
            }
            
            // Extract ORB descriptors — may remove border keypoints
            cv::Mat tracked_desc;
            auto orb_tmp = cv::ORB::create();
            orb_tmp->compute(left_undist, tracked_kpts, tracked_desc);
            
            // Detect features in right image
            auto det_right = detector.detect(right_undist);
            
            // Stereo match surviving tracked features against right image
            auto tracked_stereo = stereo_matcher.match(
                tracked_kpts, tracked_desc,
                det_right.keypoints, det_right.descriptors, calib);
            
            // Remap stereo match indices from post-compute array to original tracked_ids
            for (auto& m : tracked_stereo) {
                m.left_idx = tracked_kpts[m.left_idx].class_id;
            }
            
            // Update feature manager with stereo info for tracked features
            feature_manager.updateStereoForTracked(
                frame.timestamp_ns, tracked_ids, tracked_stereo);
            
            // --- Detect NEW features in gaps ---
            auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 15);
            auto new_stereo = stereo_matcher.match(new_det.keypoints, new_det.descriptors, det_right.keypoints, det_right.descriptors, calib);
            feature_manager.addNewFeatures(frame.timestamp_ns, new_det.keypoints, new_det.descriptors, new_stereo);

            auto current_state = optimizer.latestState();
            auto world_lms = feature_manager.getInitializedLandmarksWorld(current_state.position, current_state.rotation, calib.T_cam0_imu);
            optimizer.setLandmarks(world_lms);

            auto frame_obs = feature_manager.getObservationsForFrame(frame.timestamp_ns);
            optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), frame_obs);

            static int kf_count = 0;
            if (kf_count < 5) {
                int stereo_obs = 0;
                for (const auto& o : frame_obs) if (o.has_stereo) stereo_obs++;
                std::cerr << "[KF #" << kf_count << "] ts=" << frame.timestamp_ns
                          << " tracked_stereo=" << tracked_stereo.size()
                          << " new_det=" << new_det.keypoints.size()
                          << " new_stereo=" << new_stereo.size()
                          << " world_lms=" << world_lms.size()
                          << " frame_obs=" << frame_obs.size()
                          << " stereo_obs=" << stereo_obs
                          << " tracked=" << track_result.num_tracked
                          << "\n";
            }
            kf_count++;
            optimizer.optimize();
            
            auto latest_state = optimizer.latestState();
            traj_writer.writePose(frame.timestamp_ns, latest_state.position, latest_state.rotation);
            preintegrator.reset(latest_state.bias_gyro, latest_state.bias_accel);
            frames_since_keyframe = 0;
        } else {
            // Non-keyframe: write IMU-predicted pose for evaluation
            auto prev_state = optimizer.latestState();
            auto preint_result = preintegrator.getResult();
            if (preint_result.dt > 1e-7) {
                Eigen::Matrix3d R_prev = prev_state.rotation.toRotationMatrix();
                Eigen::Vector3d g = vio::gravity();
                double dt = preint_result.dt;
                Eigen::Vector3d pred_pos = prev_state.position + prev_state.velocity * dt
                    + 0.5 * g * dt * dt + R_prev * preint_result.delta_p;
                Eigen::Quaterniond pred_rot = prev_state.rotation * preint_result.delta_R;
                pred_rot.normalize();
                traj_writer.writePose(frame.timestamp_ns, pred_pos, pred_rot);
            }
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