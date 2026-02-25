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
#include <numeric>

#import <Foundation/Foundation.h>
#include "metal/MetalContext.h"
#include "metal/MetalUndistort.h"
#include "metal/FastDetect.h"
#include "metal/HarrisResponse.h"
#include "metal/ORBDescriptor.h"
#include "metal/StereoMatcher.h"
#include "metal/KLTTracker.h"

namespace {

// Helper: Convert Metal output to OpenCV Keypoints
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

// Helper: Grid-based NMS for uniform feature distribution
std::vector<vio::CornerPoint> gridNMS(const std::vector<vio::CornerPoint>& corners, int max_f) {
    if (corners.empty()) return {};
    std::vector<vio::CornerPoint> sorted = corners;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { 
        return a.response > b.response; 
    });
    
    std::vector<vio::CornerPoint> result;
    size_t count = std::min((size_t)max_f, sorted.size());
    for(size_t i = 0; i < count; ++i) {
        result.push_back(sorted[i]);
    }
    return result;
}

// Helper: Get GT state for initialization and viz
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

} // anon namespace

int main(int argc, char** argv) {
    @autoreleasepool {
    if (argc < 2) {
        std::cerr << "Usage: ./vio-metal <path_to_euroc_sequence> [metallib_path]\n";
        return -1;
    }

    std::string dataset_path = argv[1];
    std::string metallib_path = (argc >= 3) ? argv[2] : "shaders.metallib";

    vio::EurocLoader loader(dataset_path);
    vio::StereoCalibration calib = loader.getCalibration();
    vio::Profiler profiler;

    // --- 1. RECTIFICATION SETUP (CPU) ---
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

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_l, D_l, K_r, D_r, cv::Size(752, 480), R_rl_cv, t_rl_cv, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, cv::Size(752, 480));
    
    cv::Mat map_x_l, map_y_l, map_x_r, map_y_r;
    cv::initUndistortRectifyMap(K_l, D_l, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_l, map_y_l);
    cv::initUndistortRectifyMap(K_r, D_r, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_r, map_y_r);

    // Update calibration for VIO components
    double fx_rect = P1.at<double>(0,0);
    calib.intrinsics_left = Eigen::Vector4d(fx_rect, P1.at<double>(1,1), P1.at<double>(0,2), P1.at<double>(1,2));
    double baseline = -P2.at<double>(0,3) / fx_rect;
    calib.T_cam1_cam0 = Eigen::Matrix4d::Identity();
    calib.T_cam1_cam0(0,3) = -baseline;

    // --- 2. METAL INITIALIZATION ---
    vio::MetalContext* metal_ctx = new vio::MetalContext();
    vio::MetalUndistort* und_l = new vio::MetalUndistort(metal_ctx, map_x_l, map_y_l, 752, 480, metallib_path);
    vio::MetalUndistort* und_r = new vio::MetalUndistort(metal_ctx, map_x_r, map_y_r, 752, 480, metallib_path);
    vio::MetalKLTTracker* klt = new vio::MetalKLTTracker(metal_ctx, 752, 480, metallib_path);
    vio::MetalFastDetector* fast = new vio::MetalFastDetector(metal_ctx, 752, 480, metallib_path);
    vio::MetalHarrisResponse* harris = new vio::MetalHarrisResponse(metal_ctx, metallib_path);
    vio::MetalORBDescriptor* orb = new vio::MetalORBDescriptor(metal_ctx, metallib_path);

    // --- 3. VIO PIPELINE OBJECTS ---
    vio::FeatureManager feature_manager;
    vio::VioOptimizer optimizer(vio::VioOptimizer::Config{}, calib);
    vio::ImuPreintegrator preintegrator(vio::ImuNoiseParams{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    vio::KeyframePolicy kf_policy(vio::KeyframePolicy::Config{});
    vio::VioVisualizer visualizer;
    auto ground_truth = loader.loadGroundTruth();

    // =====================================================================
    // VIO THREAD (ASYNCHRONOUS PIPELINING)
    // =====================================================================
    std::thread vio_thread([&]() {
        @autoreleasepool {
            bool initialized = false;
            uint64_t frame_count = 0;
            uint64_t last_imu_ts = 0;
            int frames_since_kf = 0;

            while (loader.hasNext() && visualizer.isRunning()) {
                // STAGE 1: IMU PREINTEGRATION (CPU)
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

                // Load New Image
                vio::StereoFrame frame = loader.getNextStereoFrame();
                profiler.beginFrame(frame_count, frame.timestamp_ns);

                // STAGE 2: DISPATCH GPU WORK (NON-BLOCKING)
                profiler.startStage("gpu_dispatch");
                
                // Upload current frame to KLT Pyramid (Current Level)
                if (initialized) {
                    klt->buildPyramid(frame.left.data, (int)frame.left.step, false); 
                }

                // Kick off Undistort
                und_l->encodeUndistort(frame.left);  
                und_r->encodeUndistort(frame.right); 
                
                if (initialized) {
                    // Start Fused KLT Tracking (Forward + Backward)
                    auto prev_pts = feature_manager.getCurrentPoints();
                    klt->encodeTrack(prev_pts.x, prev_pts.y); 
                }
                profiler.endStage("gpu_dispatch");

                // STAGE 3: OVERLAP (CPU housekeeping while GPU works)
                auto current_gt = getGtState(ground_truth, frame.timestamp_ns);

                // STAGE 4: LATE SYNCHRONIZATION
                // This blocks only if the GPU isn't finished yet.
                profiler.startStage("gpu_sync_wait");
                metal_ctx->waitForGPU(); 
                profiler.endStage("gpu_sync_wait");

                // STAGE 5: PROCESS RESULTS (ZERO-COPY)
                if (!initialized) {
                    // Initialization Logic
                    auto raw = fast->detect(und_l->outputTexture());
                    harris->score(und_l->outputTexture(), raw);
                    auto nms = gridNMS(raw, 400);
                    
                    optimizer.initialize(getGtState(ground_truth, frame.timestamp_ns));
                    feature_manager.addNewFeatures(frame.timestamp_ns, cornersToCvKeypoints(nms), cv::Mat(), {}); 
                    
                    // Setup initial pyramid for next frame
                    klt->buildPyramid(frame.left.data, (int)frame.left.step, true);
                    initialized = true;
                } else {
                    // Normal Tracking Flow
                    auto res = klt->getResults(); 
                    feature_manager.updateTracks(frame.timestamp_ns, 
                                               feature_manager.getCurrentIds(), 
                                               res.tracked_x, res.tracked_y, res.status);

                    frames_since_kf++;
                    if (kf_policy.shouldInsertKeyframe(res.num_tracked, 0.5, frames_since_kf)) {
                        profiler.startStage("optimize");
                        optimizer.addKeyframe(frame.timestamp_ns, 
                                            preintegrator.getResult(), 
                                            feature_manager.getObservationsForFrame(frame.timestamp_ns));
                        optimizer.optimize();
                        profiler.endStage("optimize");

                        auto latest = optimizer.latestState();
                        preintegrator.reset(latest.bias_gyro, latest.bias_accel);
                        frames_since_kf = 0;
                    }
                    
                    // Current image becomes the "Previous" for the next frame
                    klt->buildPyramid(frame.left.data, (int)frame.left.step, true);
                }

                visualizer.addEstimate(optimizer.latestState().position);
                visualizer.addGroundTruth(current_gt.position);
                
                profiler.endFrame();
                frame_count++;
            }
            profiler.printSummary();
            visualizer.stop();
        }
    });

    visualizer.run();
    if (vio_thread.joinable()) vio_thread.join();

    // Cleanup
    delete und_l; delete und_r; delete klt; delete fast; delete harris; delete orb;
    delete metal_ctx;
    }
    return 0;
}