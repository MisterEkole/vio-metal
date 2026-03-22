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
#include <vector>
#include <thread>
#include <algorithm>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/MetalContext.h"
#include "metal/MetalUndistort.h"
#include "metal/FastDetect.h"
#include "metal/HarrisResponse.h"
#include "metal/ORBDescriptor.h"
#include "metal/StereoMatcher.h"

namespace {
std::vector<cv::KeyPoint> cornersToCvKeypoints(const std::vector<vio::CornerPoint>& corners) {
    std::vector<cv::KeyPoint> kpts;
    kpts.reserve(corners.size());
    for (const auto& c : corners) {
        cv::KeyPoint kp(cv::Point2f(c.position[0], c.position[1]), 31.0f);
        kp.response = c.response;
        kp.octave = (int)c.pyramid_level;
        kpts.push_back(kp);
    }
    return kpts;
}

std::vector<vio::CornerPoint> gridNMS(const std::vector<vio::CornerPoint>& corners, int max_f) {
    if (corners.empty()) return {};
    std::vector<vio::CornerPoint> sorted = corners;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { 
        return a.response > b.response; 
    });
    std::vector<vio::CornerPoint> result;
    size_t count = std::min((size_t)max_f, sorted.size());
    for(size_t i = 0; i < count; ++i) result.push_back(sorted[i]);
    return result;
}

// Convert Metal StereoMatchResult → CPU StereoMatcher::StereoMatch
std::vector<vio::StereoMatcher::StereoMatch> metalToStereoMatches(
    const std::vector<vio::StereoMatchResult>& metal_matches) {
    std::vector<vio::StereoMatcher::StereoMatch> out;
    out.reserve(metal_matches.size());
    for (const auto& m : metal_matches) {
        vio::StereoMatcher::StereoMatch sm;
        sm.left_idx = (int)m.left_idx;
        sm.right_idx = (int)m.right_idx;
        sm.disparity = m.disparity;
        sm.point_3d = Eigen::Vector3d(m.point_3d[0], m.point_3d[1], m.point_3d[2]);
        out.push_back(sm);
    }
    return out;
}

// Convert Metal ORB descriptors to cv::Mat for loop detector
cv::Mat orbOutputToCvMat(const std::vector<vio::ORBDescriptorOutput>& descs) {
    cv::Mat mat(descs.size(), 32, CV_8UC1);
    for (size_t i = 0; i < descs.size(); i++) {
        memcpy(mat.ptr(i), descs[i].desc, 32);
    }
    return mat;
}

vio::KeyframeState getGtState(const std::vector<vio::PoseStamped>& gt, uint64_t target_ts) {
    vio::KeyframeState state;
    uint64_t best_diff = UINT64_MAX;
    for (const auto& p : gt) {
        uint64_t diff = (p.timestamp_ns > target_ts) ? (p.timestamp_ns - target_ts) : (target_ts - p.timestamp_ns);
        if (diff < best_diff) {
            best_diff = diff;
            state.position = p.position; state.rotation = p.orientation;
            state.velocity = p.velocity; state.bias_gyro = p.bg; state.bias_accel = p.ba;
            state.timestamp = target_ts;
        }
    }
    return state;
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> [metallib_path] [--headless] [--quiet]\n";
        std::cerr << "  --headless  Run without visualizer window\n";
        std::cerr << "  --quiet     Suppress per-frame terminal logging\n";
        return -1;
    }
    std::string dataset_path = argv[1];
    std::string metallib_path = "shaders.metallib";
    bool headless = false;
    bool quiet = false;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--headless") headless = true;
        else if (arg == "--quiet") quiet = true;
        else metallib_path = arg;
    }

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();
    vio::Profiler profiler;

    // Stereo rectification
    cv::Mat K_l = (cv::Mat_<double>(3,3) << calib.intrinsics_left[0], 0, calib.intrinsics_left[2], 0, calib.intrinsics_left[1], calib.intrinsics_left[3], 0, 0, 1);
    cv::Mat D_l = (cv::Mat_<double>(4,1) << calib.distortion_left[0], calib.distortion_left[1], calib.distortion_left[2], calib.distortion_left[3]);
    cv::Mat K_r = (cv::Mat_<double>(3,3) << calib.intrinsics_right[0], 0, calib.intrinsics_right[2], 0, calib.intrinsics_right[1], calib.intrinsics_right[3], 0, 0, 1);
    cv::Mat D_r = (cv::Mat_<double>(4,1) << calib.distortion_right[0], calib.distortion_right[1], calib.distortion_right[2], calib.distortion_right[3]);
    
    cv::Mat R_rl_cv(3, 3, CV_64F), t_rl_cv(3, 1, CV_64F);
    Eigen::Matrix3d R_rl_eigen = calib.T_cam1_cam0.block<3,3>(0,0);
    Eigen::Vector3d t_rl_eigen = calib.T_cam1_cam0.block<3,1>(0,3);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) R_rl_cv.at<double>(r,c) = R_rl_eigen(r,c);
        t_rl_cv.at<double>(r,0) = t_rl_eigen(r);
    }

    cv::Mat R1, R2, P1, P2, Q, map_x_l, map_y_l, map_x_r, map_y_r;
    // EuRoC uses equidistant (fisheye) distortion model
    cv::fisheye::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(752, 480), R_rl_cv, t_rl_cv, R1, R2, P1, P2, Q,
                               cv::CALIB_ZERO_DISPARITY, cv::Size(752, 480));
    cv::fisheye::initUndistortRectifyMap(K_l, D_l, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_l, map_y_l);
    cv::fisheye::initUndistortRectifyMap(K_r, D_r, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_r, map_y_r);

    double fx_rect = P1.at<double>(0,0);
    calib.intrinsics_left  = Eigen::Vector4d(fx_rect, P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2));
    calib.intrinsics_right = Eigen::Vector4d(P2.at<double>(0,0), P2.at<double>(1,1), P2.at<double>(0,2), P2.at<double>(1,2));
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

    vio::MetalContext* metal_ctx = new vio::MetalContext();
    vio::MetalUndistort* und_l = new vio::MetalUndistort(metal_ctx, map_x_l, map_y_l, 752, 480, metallib_path);
    vio::MetalUndistort* und_r = new vio::MetalUndistort(metal_ctx, map_x_r, map_y_r, 752, 480, metallib_path);
    
    vio::TemporalTracker cpu_tracker;

    vio::MetalFastDetector* fast = new vio::MetalFastDetector(metal_ctx, 752, 480, metallib_path);
    vio::MetalHarrisResponse* harris = new vio::MetalHarrisResponse(metal_ctx, metallib_path);
    vio::MetalORBDescriptor* orb_gpu = new vio::MetalORBDescriptor(metal_ctx, metallib_path);
    vio::MetalStereoMatcher* stereo_gpu = new vio::MetalStereoMatcher(metal_ctx, metallib_path);

    vio::FeatureManager feature_manager;
    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::ImuPreintegrator preintegrator(vio::ImuNoiseParams{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});
    vio::LoopDetector loop_detector;
    vio::VioVisualizer* visualizer = headless ? nullptr : new vio::VioVisualizer();
    auto ground_truth = loader.loadGroundTruth();

    // Logging
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

    cv::Mat left_undist_cpu(480, 752, CV_8UC1);
    cv::Mat right_undist_cpu(480, 752, CV_8UC1); 
    cv::Mat prev_left_undist_cpu;

    auto vio_loop = [&]() {
        @autoreleasepool {
            bool initialized = false;
            uint64_t frame_count = 0;
            uint64_t last_imu_ts = 0;
            uint64_t frames_since_kf = 0;
            uint64_t total_frames = 0;

            uint64_t imu_count = 0;
            while (loader.hasNext() && (headless || visualizer->isRunning())) {
                @autoreleasepool {
                if (loader.nextIsImu()) {
                    vio::ImuSample imu = loader.getNextImuSample();
                    if (initialized) {
                        double dt = (last_imu_ts == 0) ? 0.005 : (imu.timestamp_ns - last_imu_ts) * 1e-9;
                        preintegrator.integrate(imu.gyro, imu.accel, dt);
                    }
                    last_imu_ts = imu.timestamp_ns;
                    imu_count++;
                    if (imu_count % 1000 == 0) {
                        std::printf("[imu] %llu samples processed\n", (unsigned long long)imu_count);
                        std::fflush(stdout);
                    }
                    continue;
                }

                vio::StereoFrame frame = loader.getNextStereoFrame();
                profiler.beginFrame(frame_count, frame.timestamp_ns);

                vio::KeyframeState current_gt = getGtState(ground_truth, frame.timestamp_ns);

                profiler.startStage("undistort");
                und_l->encodeUndistort(frame.left);
                und_r->encodeUndistort(frame.right);
                
                id<MTLTexture> texL = (__bridge id<MTLTexture>)und_l->outputTexture();
                id<MTLTexture> texR = (__bridge id<MTLTexture>)und_r->outputTexture();
                [texL getBytes:left_undist_cpu.data bytesPerRow:752 fromRegion:MTLRegionMake2D(0,0,752,480) mipmapLevel:0];
                [texR getBytes:right_undist_cpu.data bytesPerRow:752 fromRegion:MTLRegionMake2D(0,0,752,480) mipmapLevel:0];
                profiler.endStage("undistort");

                if (!initialized) {
                    std::printf("[init] starting detection...\n"); std::fflush(stdout);
                    profiler.startStage("detect");
                    auto raw_l = fast->detect(und_l->outputTexture());
                    harris->score(und_l->outputTexture(), raw_l);
                    auto nms_l = gridNMS(raw_l, 400);
                    auto desc_l_gpu = orb_gpu->describe(und_l->outputTexture(), nms_l);

                    auto raw_r = fast->detect(und_r->outputTexture());
                    harris->score(und_r->outputTexture(), raw_r);
                    auto nms_r = gridNMS(raw_r, 400);
                    auto desc_r_gpu = orb_gpu->describe(und_r->outputTexture(), nms_r);

                    vio::MetalStereoCalib mcalib{
                        (float)calib.intrinsics_left[0], (float)calib.intrinsics_left[1],
                        (float)calib.intrinsics_left[2], (float)calib.intrinsics_left[3],
                        (float)baseline_rect};
                    auto metal_matches = stereo_gpu->match(nms_l, desc_l_gpu, nms_r, desc_r_gpu, mcalib);
                    auto initial_matches = metalToStereoMatches(metal_matches);

                    std::vector<cv::KeyPoint> kpts_l = cornersToCvKeypoints(nms_l);
                    cv::Mat desc_l = orbOutputToCvMat(desc_l_gpu);

                    optimizer.initialize(current_gt);
                    feature_manager.addNewFeatures(frame.timestamp_ns, kpts_l, desc_l, initial_matches);

                    // KF0 observations anchor landmarks at the fixed first frame
                    optimizer.addObservations(frame.timestamp_ns, feature_manager.getObservationsForFrame(frame.timestamp_ns));
                    auto lm_world_init = feature_manager.getInitializedLandmarksWorld(
                        current_gt.position, current_gt.rotation, calib.T_cam0_imu);
                    optimizer.setLandmarks(lm_world_init);

                    profiler.setCount("features_detected", (int)kpts_l.size());
                    profiler.setCount("stereo_matches", (int)initial_matches.size());
                    profiler.setCount("landmarks_initialized", feature_manager.numInitializedLandmarks());

                    prev_left_undist_cpu = left_undist_cpu.clone();
                    initialized = true;
                    profiler.endStage("detect");
                    std::printf("[init] %d features, %d stereo, %d landmarks\n",
                        (int)kpts_l.size(), (int)initial_matches.size(),
                        feature_manager.numInitializedLandmarks());
                } else {
                    std::printf("."); std::fflush(stdout);
                    profiler.startStage("temporal_track");
                    std::vector<cv::Point2f> pts_to_track = feature_manager.getCurrentPoints();
                    auto res = cpu_tracker.track(prev_left_undist_cpu, left_undist_cpu, pts_to_track);
                    feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), res.tracked_points, res.status);
                    profiler.setCount("tracked", res.num_tracked);
                    profiler.endStage("temporal_track");

                    if (!quiet) {
                        std::printf("[%4llu] tracked: %d/%d  landmarks: %d\n",
                            (unsigned long long)total_frames, res.num_tracked,
                            (int)pts_to_track.size(), feature_manager.numInitializedLandmarks());
                        std::fflush(stdout);
                    }

                    double parallax = vio::TemporalTracker::averageParallax(
                        pts_to_track, res.tracked_points, res.status, calib.intrinsics_left[0]);
                    frames_since_kf++;
                    if (kf_policy.shouldInsertKeyframe(res.num_tracked, parallax, (int)frames_since_kf)) {
                        profiler.startStage("stereo_retrack");
                        std::vector<cv::Point2f> tracked_pts = feature_manager.getCurrentPoints();
                        std::vector<uint64_t> tracked_ids = feature_manager.getCurrentIds();

                        if (!tracked_pts.empty()) {
                            // Convert tracked points to CornerPoints for Metal ORB
                            std::vector<vio::CornerPoint> tracked_corners;
                            tracked_corners.reserve(tracked_pts.size());
                            for (const auto& pt : tracked_pts) {
                                vio::CornerPoint cp;
                                cp.position[0] = pt.x; cp.position[1] = pt.y;
                                cp.response = 1.0f; cp.pyramid_level = 0;
                                tracked_corners.push_back(cp);
                            }
                            auto tracked_desc_gpu = orb_gpu->describe(und_l->outputTexture(), tracked_corners);

                            // Detect + describe right image with Metal
                            auto raw_r = fast->detect(und_r->outputTexture());
                            harris->score(und_r->outputTexture(), raw_r);
                            auto nms_r = gridNMS(raw_r, 400);
                            auto desc_r_gpu = orb_gpu->describe(und_r->outputTexture(), nms_r);

                            vio::MetalStereoCalib mcalib{
                                (float)calib.intrinsics_left[0], (float)calib.intrinsics_left[1],
                                (float)calib.intrinsics_left[2], (float)calib.intrinsics_left[3],
                                (float)baseline_rect};
                            auto metal_retrack = stereo_gpu->match(tracked_corners, tracked_desc_gpu, nms_r, desc_r_gpu, mcalib);
                            auto retrack_matches = metalToStereoMatches(metal_retrack);
                            feature_manager.updateStereoForTracked(frame.timestamp_ns, tracked_ids, retrack_matches);

                            profiler.setCount("stereo_retracked", (int)retrack_matches.size());
                            if (!quiet) {
                                std::printf("  [stereo retrack] tracked: %d  re-matched: %d\n",
                                    (int)tracked_pts.size(), (int)retrack_matches.size());
                            }
                        }
                        profiler.endStage("stereo_retrack");

                        profiler.startStage("detect");
                        auto raw_new = fast->detect(und_l->outputTexture());
                        harris->score(und_l->outputTexture(), raw_new);
                        auto nms_new = gridNMS(raw_new, 300);

                        // Filter detections near existing tracks
                        {
                            auto existing = feature_manager.getCurrentPoints();
                            if (!existing.empty()) {
                                cv::Mat mask = cv::Mat::ones(480, 752, CV_8UC1) * 255;
                                for (const auto& pt : existing)
                                    cv::circle(mask, pt, 10, cv::Scalar(0), -1);
                                std::vector<vio::CornerPoint> filtered;
                                for (const auto& cp : nms_new) {
                                    int px = (int)cp.position[0], py = (int)cp.position[1];
                                    if (px >= 0 && px < 752 && py >= 0 && py < 480 && mask.at<uint8_t>(py, px) > 0)
                                        filtered.push_back(cp);
                                }
                                nms_new = std::move(filtered);
                            }
                        }
                        profiler.endStage("detect");

                        profiler.startStage("stereo_match");
                        auto desc_new_l = orb_gpu->describe(und_l->outputTexture(), nms_new);

                        auto raw_r_new = fast->detect(und_r->outputTexture());
                        harris->score(und_r->outputTexture(), raw_r_new);
                        auto nms_r_new = gridNMS(raw_r_new, 400);
                        auto desc_new_r = orb_gpu->describe(und_r->outputTexture(), nms_r_new);

                        vio::MetalStereoCalib mcalib_new{
                            (float)calib.intrinsics_left[0], (float)calib.intrinsics_left[1],
                            (float)calib.intrinsics_left[2], (float)calib.intrinsics_left[3],
                            (float)baseline_rect};
                        auto metal_new_matches = stereo_gpu->match(nms_new, desc_new_l, nms_r_new, desc_new_r, mcalib_new);
                        auto matches = metalToStereoMatches(metal_new_matches);

                        std::vector<cv::KeyPoint> kpts_l = cornersToCvKeypoints(nms_new);
                        cv::Mat desc_l = orbOutputToCvMat(desc_new_l);
                        feature_manager.addNewFeatures(frame.timestamp_ns, kpts_l, desc_l, matches);
                        profiler.setCount("features_detected", (int)kpts_l.size());
                        profiler.setCount("stereo_matches", (int)matches.size());
                        profiler.setCount("landmarks_initialized", feature_manager.numInitializedLandmarks());
                        profiler.endStage("stereo_match");

                        profiler.startStage("optimize");
                        auto preint_result = preintegrator.getResult();
                        optimizer.addKeyframe(frame.timestamp_ns, preint_result, feature_manager.getObservationsForFrame(frame.timestamp_ns));

                        if (!quiet) {
                            auto imu_pred = optimizer.latestState(); // IMU-propagated (before optimize)
                            auto gt_kf = getGtState(ground_truth, frame.timestamp_ns);
                            double pred_err = (imu_pred.position - gt_kf.position).norm();
                            double vel_err = (imu_pred.velocity - gt_kf.velocity).norm();
                            Eigen::Quaterniond dq = gt_kf.rotation.inverse() * imu_pred.rotation;
                            if (dq.w() < 0) dq.coeffs() *= -1;
                            double rot_err_deg = 2.0 * std::asin(dq.vec().norm()) * 180.0 / M_PI;
                            std::printf("  [IMU pred] pos_err: %.4fm  vel_err: %.4fm/s  rot_err: %.2fdeg  dt: %.4fs  bg: (%.5f, %.5f, %.5f)  ba: (%.5f, %.5f, %.5f)\n",
                                pred_err, vel_err, rot_err_deg, preint_result.dt,
                                imu_pred.bias_gyro.x(), imu_pred.bias_gyro.y(), imu_pred.bias_gyro.z(),
                                imu_pred.bias_accel.x(), imu_pred.bias_accel.y(), imu_pred.bias_accel.z());
                        }

                        auto lm_world = feature_manager.getInitializedLandmarksWorld(
                            optimizer.latestState().position,
                            optimizer.latestState().rotation,
                            calib.T_cam0_imu);
                        optimizer.setLandmarks(lm_world);
                        profiler.setCount("landmarks", (int)lm_world.size());
                        optimizer.optimize();
                        profiler.endStage("optimize");

                        auto latest_state = optimizer.latestState();

                        if (!quiet) {
                            auto gt_kf = getGtState(ground_truth, frame.timestamp_ns);
                            double opt_err = (latest_state.position - gt_kf.position).norm();
                            std::printf("  [OPT done] pos_err: %.4fm  window: %d  cost: %.1f -> %.1f  %s\n",
                                opt_err, optimizer.windowSize(),
                                optimizer.lastSummary().initial_cost, optimizer.lastSummary().final_cost,
                                optimizer.lastSummary().success ? "CONV" : "FAIL");
                        }

                        {
                            vio::KeyframeDescriptorEntry kf_entry;
                            kf_entry.timestamp = frame.timestamp_ns;
                            kf_entry.descriptors = desc_l.clone();
                            kf_entry.keypoints = kpts_l;
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
                        frames_since_kf = 0;
                    }
                    prev_left_undist_cpu = left_undist_cpu.clone();
                }

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
                        double pos_err = (latest.position - current_gt.position).norm();
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
                    visualizer->addEstimate(latest.position);
                    visualizer->addGroundTruth(current_gt.position);
                }

                profiler.endFrame();
                frame_count++;
                } // @autoreleasepool per iteration
            }
            traj_writer.flush();
            cost_log.close();
            profiler.writeCSV(results_dir + "/timing_" + run_tag + ".csv");
            profiler.printSummary();
            if (visualizer) visualizer->stop();
        }
    };

    if (headless) {
        vio_loop();
    } else {
        std::thread vio_thread(vio_loop);
        visualizer->run();
        if (vio_thread.joinable()) vio_thread.join();
    }

    delete visualizer; delete und_l; delete und_r; delete fast; delete harris; delete orb_gpu; delete stereo_gpu; delete metal_ctx;
    }
    return 0;
}