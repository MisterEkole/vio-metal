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
#include "vision/LoopDetector.h"
#include "visualizer.h"

#include <iostream>
#include <fstream>
#include <chrono>
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
        std::cerr << "Usage: " << argv[0] << " <dataset_path> [--headless] [--quiet]\n";
        std::cerr << "  --headless  Run without visualizer window\n";
        std::cerr << "  --quiet     Suppress per-frame terminal logging\n";
        return -1;
    }

    std::string dataset_path = argv[1];
    bool headless = false;
    bool quiet = false;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--headless") headless = true;
        else if (arg == "--quiet") quiet = true;
    }

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();
    vio::Profiler profiler;

    cv::Mat K_left = (cv::Mat_<double>(3,3) << calib.intrinsics_left[0], 0, calib.intrinsics_left[2], 0, calib.intrinsics_left[1], calib.intrinsics_left[3], 0, 0, 1);
    cv::Mat dist_left = (cv::Mat_<double>(4,1) << calib.distortion_left[0], calib.distortion_left[1], calib.distortion_left[2], calib.distortion_left[3]);
    cv::Mat K_right = (cv::Mat_<double>(3,3) << calib.intrinsics_right[0], 0, calib.intrinsics_right[2], 0, calib.intrinsics_right[1], calib.intrinsics_right[3], 0, 0, 1);
    cv::Mat dist_right = (cv::Mat_<double>(4,1) << calib.distortion_right[0], calib.distortion_right[1], calib.distortion_right[2], calib.distortion_right[3]);

    Eigen::Matrix3d R_rl_eigen = calib.T_cam1_cam0.block<3,3>(0,0);
    Eigen::Vector3d t_rl_eigen = calib.T_cam1_cam0.block<3,1>(0,3);
    cv::Mat R_rl_cv(3, 3, CV_64F), t_rl_cv(3, 1, CV_64F);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) R_rl_cv.at<double>(r,c) = R_rl_eigen(r,c);
        t_rl_cv.at<double>(r,0) = t_rl_eigen(r);
    }

    cv::Mat R1, R2, P1, P2, Q;
    // EuRoC uses equidistant (fisheye) distortion model
    cv::fisheye::stereoRectify(K_left, dist_left, K_right, dist_right,
                      cv::Size(752, 480), R_rl_cv, t_rl_cv,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, cv::Size(752, 480));

    cv::Mat map_x_left, map_y_left, map_x_right, map_y_right;
    cv::fisheye::initUndistortRectifyMap(K_left, dist_left, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
    cv::fisheye::initUndistortRectifyMap(K_right, dist_right, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);

    // Update calibration for rectified frames
    double fx_rect = P1.at<double>(0,0);
    calib.intrinsics_left  = Eigen::Vector4d(fx_rect, P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2));
    double baseline_rect = -P2.at<double>(0,3) / fx_rect;
    calib.T_cam1_cam0 = Eigen::Matrix4d::Identity();
    calib.T_cam1_cam0(0,3) = -baseline_rect;

    // Apply rectification rotation R1 to camera-IMU extrinsic
    {
        Eigen::Matrix3d R1_eigen;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R1_eigen(r, c) = R1.at<double>(r, c);
        Eigen::Matrix4d T_rect = Eigen::Matrix4d::Identity();
        T_rect.block<3,3>(0,0) = R1_eigen;
        calib.T_cam0_imu = T_rect * calib.T_cam0_imu;
    }

    vio::FeatureDetector detector(vio::FeatureDetector::Config{});
    vio::StereoMatcher stereo_matcher(vio::StereoMatcher::Config{});
    vio::TemporalTracker tracker(vio::TemporalTracker::Config{});
    vio::FeatureManager feature_manager;
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});
    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::ImuPreintegrator preintegrator(vio::ImuNoiseParams{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    vio::LoopDetector loop_detector;
    vio::VioVisualizer* visualizer = headless ? nullptr : new vio::VioVisualizer();
    auto ground_truth = loader.loadGroundTruth();

    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char run_ts[20];
    std::strftime(run_ts, sizeof(run_ts), "%Y%m%d_%H%M%S", std::localtime(&now_t));
    std::string run_tag(run_ts);

    std::string results_dir = "results/configs";
    system(("mkdir -p " + results_dir).c_str());
    system("mkdir -p results/trajectories");
    vio::TrajectoryWriter traj_writer("results/trajectories/estimated_" + run_tag + ".txt");
    std::ofstream cost_log(results_dir + "/cost_log_" + run_tag + ".csv");
    cost_log << "timestamp,initial_cost,final_cost,iterations,num_residuals,num_landmarks,converged\n";
    std::cout << "[Run] Tag: " << run_tag << "\n";

    auto vio_loop = [&]() {
        bool pipeline_initialized = false;
        cv::Mat prev_left_undistorted;
        uint64_t frame_count = 0;
        uint64_t last_imu_ts = 0;
        int frames_since_keyframe = 0;
        uint64_t total_frames = 0;

        while (loader.hasNext() && (headless || visualizer->isRunning())) {
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

                profiler.startStage("detect");
                auto det = detector.detect(left_undist);
                auto det_right = detector.detect(right_undist);
                profiler.endStage("detect");

                profiler.startStage("stereo_match");
                auto stereo_matches = stereo_matcher.match(det.keypoints, det.descriptors, det_right.keypoints, det_right.descriptors, calib);
                profiler.endStage("stereo_match");
                
                feature_manager.addNewFeatures(frame.timestamp_ns, det.keypoints, det.descriptors, stereo_matches);

                // KF0 observations anchor landmarks at the fixed first frame
                optimizer.addObservations(frame.timestamp_ns, feature_manager.getObservationsForFrame(frame.timestamp_ns));
                auto lm_world_init = feature_manager.getInitializedLandmarksWorld(
                    init_state.position, init_state.rotation, calib.T_cam0_imu);
                optimizer.setLandmarks(lm_world_init);

                pipeline_initialized = true;
                current_est_pos = init_state.position;
                pose_valid = true;
            } else {
                profiler.startStage("temporal_track");
                auto prev_points = feature_manager.getCurrentPoints();
                auto track_result = tracker.track(prev_left_undistorted, left_undist, prev_points);
                feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), track_result.tracked_points, track_result.status);
                profiler.setCount("tracked", track_result.num_tracked);
                profiler.endStage("temporal_track");

                if (!quiet) {
                    std::printf("[%4llu] tracked: %d/%d  landmarks: %d\n",
                        (unsigned long long)total_frames, track_result.num_tracked,
                        (int)prev_points.size(), feature_manager.numInitializedLandmarks());
                    std::fflush(stdout);
                }

                double parallax = vio::TemporalTracker::averageParallax(prev_points, track_result.tracked_points, track_result.status, calib.intrinsics_left[0]);
                frames_since_keyframe++;

                if (kf_policy.shouldInsertKeyframe(track_result.num_tracked, parallax, frames_since_keyframe)) {
                    profiler.startStage("detect");
                    auto det_right = detector.detect(right_undist);
                    profiler.endStage("detect");

                    profiler.startStage("describe");
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
                    profiler.setCount("stereo_retracked", (int)tracked_stereo.size());
                    profiler.endStage("stereo_match");

                    if (!quiet) {
                        std::printf("  [stereo retrack] tracked: %d  re-matched: %d\n",
                            (int)tracked_kpts.size(), (int)tracked_stereo.size());
                        std::fflush(stdout);
                    }

                    profiler.startStage("detect");
                    auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 10);
                    profiler.endStage("detect");
                    
                    profiler.startStage("stereo_match");
                    auto new_stereo = stereo_matcher.match(new_det.keypoints, new_det.descriptors, det_right.keypoints, det_right.descriptors, calib);
                    feature_manager.addNewFeatures(frame.timestamp_ns, new_det.keypoints, new_det.descriptors, new_stereo);
                    profiler.setCount("features_detected", (int)new_det.keypoints.size());
                    profiler.setCount("stereo_matches", (int)new_stereo.size());
                    profiler.setCount("landmarks_initialized", feature_manager.numInitializedLandmarks());
                    profiler.endStage("stereo_match");

                    profiler.startStage("optimize");
                    optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), feature_manager.getObservationsForFrame(frame.timestamp_ns));
                    auto lm_world = feature_manager.getInitializedLandmarksWorld(
                        optimizer.latestState().position,
                        optimizer.latestState().rotation,
                        calib.T_cam0_imu);
                    optimizer.setLandmarks(lm_world);
                    profiler.setCount("landmarks", (int)lm_world.size());
                    optimizer.optimize();
                    profiler.endStage("optimize");
                    
                    auto latest_state = optimizer.latestState();

                    // Loop closure
                    {
                        vio::KeyframeDescriptorEntry kf_entry;
                        kf_entry.timestamp = frame.timestamp_ns;
                        if (!tracked_desc.empty() && !new_det.descriptors.empty()) {
                            cv::vconcat(tracked_desc, new_det.descriptors, kf_entry.descriptors);
                        } else if (!tracked_desc.empty()) {
                            kf_entry.descriptors = tracked_desc;
                        } else {
                            kf_entry.descriptors = new_det.descriptors;
                        }
                        kf_entry.keypoints = tracked_kpts;
                        for (const auto& kp : new_det.keypoints)
                            kf_entry.keypoints.push_back(kp);
                        kf_entry.feature_ids = feature_manager.getCurrentIds();
                        kf_entry.landmarks_world = lm_world;

                        loop_detector.addKeyframe(kf_entry);

                        auto loop = loop_detector.detectLoop(kf_entry,
                            calib.intrinsics_left[0], calib.intrinsics_left[1],
                            calib.intrinsics_left[2], calib.intrinsics_left[3],
                            calib.T_cam0_imu);

                        if (loop.valid) {
                            vio::LoopConstraint lc;
                            lc.timestamp_i = loop.match_timestamp;
                            lc.timestamp_j = loop.query_timestamp;

                            Eigen::Vector3d p_match = Eigen::Vector3d::Zero();
                            Eigen::Quaterniond q_match = Eigen::Quaterniond::Identity();
                            bool found = false;
                            for (const auto& kf : optimizer.window()) {
                                if (kf.timestamp == loop.match_timestamp) {
                                    p_match = kf.position;
                                    q_match = kf.rotation;
                                    found = true;
                                    break;
                                }
                            }

                            if (found) {
                                lc.relative_position = q_match.inverse() * (loop.relative_position - p_match);
                                lc.relative_rotation = q_match.inverse() * loop.relative_rotation;
                                lc.sqrt_info = Eigen::Matrix<double, 6, 6>::Identity() * 10.0;
                                optimizer.addLoopConstraint(lc);
                                optimizer.optimize();
                                latest_state = optimizer.latestState();
                            }
                        }
                    }

                    preintegrator.reset(latest_state.bias_gyro, latest_state.bias_accel);
                    frames_since_keyframe = 0;
                    current_est_pos = latest_state.position;
                    pose_valid = true;
                } else {
                    current_est_pos = optimizer.latestState().position;
                    pose_valid = true;
                }
            }

            prev_left_undistorted = left_undist.clone();
            if (pose_valid) {
                auto latest = optimizer.latestState();
                traj_writer.writePose(frame.timestamp_ns, latest.position, latest.rotation);
                total_frames++;

                auto& cs = optimizer.lastSummary();
                if (cs.iterations > 0) {
                    cost_log << frame.timestamp_ns << ","
                             << cs.initial_cost << "," << cs.final_cost << ","
                             << cs.iterations << "," << cs.num_residuals << ","
                             << cs.num_landmarks << "," << (cs.success ? 1 : 0) << "\n";

                    if (!quiet) {
                        auto gt_state = getInitialState(ground_truth, frame.timestamp_ns);
                        double pos_err = (latest.position - gt_state.position).norm();
                        std::printf("[%4llu] cost: %.1f -> %.1f  iter: %d  lm: %d  res: %d  %s  pos: (%.2f, %.2f, %.2f)  err: %.3fm\n",
                            (unsigned long long)total_frames,
                            cs.initial_cost, cs.final_cost,
                            cs.iterations, cs.num_landmarks, cs.num_residuals,
                            cs.success ? "CONV" : "FAIL",
                            latest.position.x(), latest.position.y(), latest.position.z(),
                            pos_err);
                    }
                }

                if (visualizer) {
                    visualizer->addEstimate(current_est_pos);
                    visualizer->addGroundTruth(getInitialState(ground_truth, frame.timestamp_ns).position);
                }
            }

            profiler.endFrame();
            frame_count++;
        }
        traj_writer.flush();
        cost_log.close();
        profiler.writeCSV(results_dir + "/timing_" + run_tag + ".csv");
        profiler.printSummary();
        if (visualizer) visualizer->stop();
    };

    if (headless) {
        vio_loop();
    } else {
        std::thread vio_thread(vio_loop);
        visualizer->run();
        if (vio_thread.joinable()) vio_thread.join();
    }

    delete visualizer;
    return 0;
}