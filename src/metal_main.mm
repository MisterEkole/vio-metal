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
#include <vector>
#include <thread>
#include <algorithm>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/MetalContext.h"
#include "metal/MetalUndistort.h"
#include "metal/FastDetect.h"
#include "metal/HarrisResponse.h"

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
    if (argc < 2) return -1;
    std::string dataset_path = argv[1];
    std::string metallib_path = (argc >= 3) ? argv[2] : "shaders.metallib";

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();
    vio::Profiler profiler;

    // --- RECTIFICATION SETUP ---
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
    cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(752, 480), R_rl_cv, t_rl_cv, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(K_l, D_l, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_l, map_y_l);
    cv::initUndistortRectifyMap(K_r, D_r, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_r, map_y_r);

    // --- INITIALIZATION ---
    vio::MetalContext* metal_ctx = new vio::MetalContext();
    vio::MetalUndistort* und_l = new vio::MetalUndistort(metal_ctx, map_x_l, map_y_l, 752, 480, metallib_path);
    vio::MetalUndistort* und_r = new vio::MetalUndistort(metal_ctx, map_x_r, map_y_r, 752, 480, metallib_path);
    
    vio::TemporalTracker cpu_tracker; 
    vio::StereoMatcher cpu_stereo; 
    cv::Ptr<cv::ORB> orb_extractor = cv::ORB::create(1000); 

    vio::MetalFastDetector* fast = new vio::MetalFastDetector(metal_ctx, 752, 480, metallib_path);
    vio::MetalHarrisResponse* harris = new vio::MetalHarrisResponse(metal_ctx, metallib_path);

    vio::FeatureManager feature_manager;
    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::ImuPreintegrator preintegrator(vio::ImuNoiseParams{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});
    vio::VioVisualizer visualizer;
    auto ground_truth = loader.loadGroundTruth();

    cv::Mat left_undist_cpu(480, 752, CV_8UC1);
    cv::Mat right_undist_cpu(480, 752, CV_8UC1); 
    cv::Mat prev_left_undist_cpu;

    std::thread vio_thread([&]() {
        @autoreleasepool {
            bool initialized = false;
            uint64_t frame_count = 0;
            uint64_t last_imu_ts = 0;
            uint64_t frames_since_kf = 0;

            while (loader.hasNext() && visualizer.isRunning()) {
                if (loader.nextIsImu()) {
                    profiler.startStage("preintegration");
                    vio::ImuSample imu = loader.getNextImuSample();
                    if (initialized) {
                        double dt = (last_imu_ts == 0) ? 0.005 : (imu.timestamp_ns - last_imu_ts) * 1e-9;
                        preintegrator.integrate(imu.gyro, imu.accel, dt);
                    }
                    last_imu_ts = imu.timestamp_ns;
                    profiler.endStage("preintegration");
                    continue;
                }

                vio::StereoFrame frame = loader.getNextStereoFrame();
                profiler.beginFrame(frame_count, frame.timestamp_ns);

                
                vio::KeyframeState current_gt = getGtState(ground_truth, frame.timestamp_ns);

                profiler.startStage("undistort");
                und_l->encodeUndistort(frame.left);
                und_r->encodeUndistort(frame.right);
                metal_ctx->waitForLastBuffer(); 
                
                id<MTLTexture> texL = (__bridge id<MTLTexture>)und_l->outputTexture();
                id<MTLTexture> texR = (__bridge id<MTLTexture>)und_r->outputTexture();
                [texL getBytes:left_undist_cpu.data bytesPerRow:752 fromRegion:MTLRegionMake2D(0,0,752,480) mipmapLevel:0];
                [texR getBytes:right_undist_cpu.data bytesPerRow:752 fromRegion:MTLRegionMake2D(0,0,752,480) mipmapLevel:0];
                profiler.endStage("undistort");

                if (!initialized) {
                    profiler.startStage("detect");
                    auto raw = fast->detect(und_l->outputTexture());
                    harris->score(und_l->outputTexture(), raw);
                    metal_ctx->waitForLastBuffer();
                    auto nms = gridNMS(raw, 400);
                    std::vector<cv::KeyPoint> kpts_l = cornersToCvKeypoints(nms);
                    
                    cv::Mat desc_l, desc_r;
                    std::vector<cv::KeyPoint> kpts_r_all; 
                    orb_extractor->detectAndCompute(right_undist_cpu, cv::noArray(), kpts_r_all, desc_r);
                    orb_extractor->compute(left_undist_cpu, kpts_l, desc_l);
                    
                    auto initial_matches = cpu_stereo.match(kpts_l, desc_l, kpts_r_all, desc_r, calib);
                    
                    optimizer.initialize(current_gt); // Use GT for initialization
                    feature_manager.addNewFeatures(frame.timestamp_ns, kpts_l, desc_l, initial_matches); 
                    
                    prev_left_undist_cpu = left_undist_cpu.clone();
                    initialized = true;
                    profiler.endStage("detect");
                } else {
                    profiler.startStage("temporal_track");
                    std::vector<cv::Point2f> pts_to_track = feature_manager.getCurrentPoints();
                    auto res = cpu_tracker.track(prev_left_undist_cpu, left_undist_cpu, pts_to_track);
                    feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(), res.tracked_points, res.status);
                    profiler.endStage("temporal_track");

                    frames_since_kf++;
                    if (kf_policy.shouldInsertKeyframe(res.num_tracked, 0.5, (int)frames_since_kf)) {
                        profiler.startStage("detect");
                        auto raw = fast->detect(und_l->outputTexture());
                        harris->score(und_l->outputTexture(), raw);
                        metal_ctx->waitForLastBuffer();
                        auto nms = gridNMS(raw, 150);
                        std::vector<cv::KeyPoint> kpts_l = cornersToCvKeypoints(nms);
                        profiler.endStage("detect");

                        profiler.startStage("stereo_match");
                        cv::Mat desc_l, desc_r;
                        std::vector<cv::KeyPoint> kpts_r_all;
                        orb_extractor->detectAndCompute(right_undist_cpu, cv::noArray(), kpts_r_all, desc_r);
                        orb_extractor->compute(left_undist_cpu, kpts_l, desc_l);
                        
                        auto matches = cpu_stereo.match(kpts_l, desc_l, kpts_r_all, desc_r, calib);
                        feature_manager.addNewFeatures(frame.timestamp_ns, kpts_l, desc_l, matches);
                        profiler.endStage("stereo_match");

                        profiler.startStage("optimize");
                        optimizer.addKeyframe(frame.timestamp_ns, preintegrator.getResult(), feature_manager.getObservationsForFrame(frame.timestamp_ns));
                        optimizer.optimize();
                        profiler.endStage("optimize");

                        preintegrator.reset(optimizer.latestState().bias_gyro, optimizer.latestState().bias_accel);
                        frames_since_kf = 0;
                    }
                    prev_left_undist_cpu = left_undist_cpu.clone();
                }

                // --- PLOTTING LOGIC ---
                visualizer.addEstimate(optimizer.latestState().position);
                visualizer.addGroundTruth(current_gt.position); // Feed GT and estimates to visualizer
                
                profiler.endFrame();
                frame_count++;
            }
            profiler.printSummary();
            visualizer.stop();
        }
    });

    visualizer.run();
    if (vio_thread.joinable()) vio_thread.join();

    delete und_l; delete und_r; delete fast; delete harris; delete metal_ctx;
    }
    return 0;
}