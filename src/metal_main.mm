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
#include <algorithm>
#include <numeric>
#include <thread>

#import <Foundation/Foundation.h>
#include "metal/MetalContext.h"
#include "metal/MetalUndistort.h"

// --- Metal GPU kernels (Phase 2) ---
#include "metal/FastDetect.h"
#include "metal/HarrisResponse.h"
#include "metal/ORBDescriptor.h"
#include "metal/StereoMatcher.h"
#include "metal/KLTTracker.h"

namespace {

std::vector<cv::KeyPoint> cornersToCvKeypoints(const std::vector<vio::CornerPoint>& corners) {
    std::vector<cv::KeyPoint> kpts;
    kpts.reserve(corners.size());
    for (size_t i = 0; i < corners.size(); i++) {
        cv::KeyPoint kp;
        kp.pt.x = corners[i].position[0];
        kp.pt.y = corners[i].position[1];
        kp.response = corners[i].response;
        kp.size = 31.0f;
        kp.octave = (int)corners[i].pyramid_level;
        kp.class_id = (int)i;
        kpts.push_back(kp);
    }
    return kpts;
}

cv::Mat orbOutputToCvMat(const std::vector<vio::ORBDescriptorOutput>& descs) {
    cv::Mat mat((int)descs.size(), 32, CV_8UC1);
    for (size_t i = 0; i < descs.size(); i++) {
        memcpy(mat.ptr((int)i), descs[i].desc, 32);
    }
    return mat;
}

std::vector<vio::StereoMatcher::StereoMatch> metalToVioStereoMatches(
    const std::vector<vio::StereoMatchResult>& metal_matches)
{
    std::vector<vio::StereoMatcher::StereoMatch> result;
    result.reserve(metal_matches.size());
    for (const auto& m : metal_matches) {
        vio::StereoMatcher::StereoMatch sm;
        sm.left_idx = m.left_idx;
        sm.right_idx = m.right_idx;
        sm.point_3d = Eigen::Vector3d(m.point_3d[0], m.point_3d[1], m.point_3d[2]);
        result.push_back(sm);
    }
    return result;
}

std::vector<vio::CornerPoint> gridNMS(const std::vector<vio::CornerPoint>& corners,
                                       int image_w, int image_h,
                                       int max_features, int grid_cols = 8, int grid_rows = 5,
                                       float min_distance = 8.0f)
{
    if (corners.empty()) return {};

    int cell_w = image_w / grid_cols;
    int cell_h = image_h / grid_rows;
    int per_cell = std::max(1, max_features / (grid_cols * grid_rows));

    std::vector<size_t> idx(corners.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return corners[a].response > corners[b].response;
    });

    std::vector<std::vector<size_t>> cells(grid_cols * grid_rows);
    for (size_t si = 0; si < idx.size(); si++) {
        const auto& c = corners[idx[si]];
        int gx = std::min((int)(c.position[0] / cell_w), grid_cols - 1);
        int gy = std::min((int)(c.position[1] / cell_h), grid_rows - 1);
        auto& cell = cells[gy * grid_cols + gx];
        if ((int)cell.size() < per_cell * 3)
            cell.push_back(idx[si]);
    }

    float min_dist_sq = min_distance * min_distance;
    std::vector<vio::CornerPoint> result;
    result.reserve(max_features);

    for (const auto& cell : cells) {
        int kept = 0;
        std::vector<bool> suppressed(cell.size(), false);
        for (size_t i = 0; i < cell.size() && kept < per_cell; i++) {
            if (suppressed[i]) continue;
            result.push_back(corners[cell[i]]);
            kept++;
            for (size_t j = i + 1; j < cell.size(); j++) {
                float dx = corners[cell[i]].position[0] - corners[cell[j]].position[0];
                float dy = corners[cell[i]].position[1] - corners[cell[j]].position[1];
                if (dx * dx + dy * dy < min_dist_sq) suppressed[j] = true;
            }
        }
    }

    return result;
}

std::vector<vio::CornerPoint> filterByMask(const std::vector<vio::CornerPoint>& corners,
                                            const std::vector<cv::Point2f>& existing_points,
                                            int mask_radius)
{
    if (existing_points.empty()) return corners;
    float r2 = (float)(mask_radius * mask_radius);

    std::vector<vio::CornerPoint> filtered;
    filtered.reserve(corners.size());
    for (const auto& c : corners) {
        bool too_close = false;
        for (const auto& ep : existing_points) {
            float dx = c.position[0] - ep.x;
            float dy = c.position[1] - ep.y;
            if (dx * dx + dy * dy < r2) { too_close = true; break; }
        }
        if (!too_close) filtered.push_back(c);
    }
    return filtered;
}

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

struct MetalDetectionResult {
    std::vector<vio::CornerPoint> corners;
    std::vector<vio::ORBDescriptorOutput> descs;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

MetalDetectionResult metalDetect(vio::MetalFastDetector& fast,
                                 vio::MetalHarrisResponse& harris,
                                 vio::MetalORBDescriptor& orb,
                                 void* image_texture,
                                 int width, int height,
                                 int max_features = 400,
                                 const std::vector<cv::Point2f>* existing_points = nullptr,
                                 int mask_radius = 15)
{
    MetalDetectionResult result;

    auto raw_corners = fast.detect(image_texture);
    harris.score(image_texture, raw_corners);
    auto nms_corners = gridNMS(raw_corners, width, height, max_features);

    if (existing_points && !existing_points->empty()) {
        nms_corners = filterByMask(nms_corners, *existing_points, mask_radius);
    }

    if (nms_corners.empty()) return result;

    result.descs = orb.describe(image_texture, nms_corners);
    result.corners = nms_corners;
    result.keypoints = cornersToCvKeypoints(nms_corners);
    result.descriptors = orbOutputToCvMat(result.descs);

    return result;
}

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
    cv::stereoRectify(K_left, dist_left, K_right, dist_right,
                      cv::Size(752, 480), R_rl_cv, t_rl_cv,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, cv::Size(752, 480));

    cv::Mat map_x_left, map_y_left, map_x_right, map_y_right;
    cv::initUndistortRectifyMap(K_left, dist_left, R1, P1, cv::Size(752, 480), CV_32FC1, map_x_left, map_y_left);
    cv::initUndistortRectifyMap(K_right, dist_right, R2, P2, cv::Size(752, 480), CV_32FC1, map_x_right, map_y_right);

    double fx_rect = P1.at<double>(0,0);
    double fy_rect = P1.at<double>(1,1);
    double cx_rect = P1.at<double>(0,2);
    double cy_rect = P1.at<double>(1,2);
    calib.intrinsics_left  = Eigen::Vector4d(fx_rect, fy_rect, cx_rect, cy_rect);
    calib.intrinsics_right = Eigen::Vector4d(P2.at<double>(0,0), P2.at<double>(1,1),
                                              P2.at<double>(0,2), P2.at<double>(1,2));

    double baseline_rect = -P2.at<double>(0,3) / fx_rect;
    Eigen::Matrix4d T_cam1_cam0_rect = Eigen::Matrix4d::Identity();
    T_cam1_cam0_rect(0,3) = -baseline_rect;
    calib.T_cam1_cam0 = T_cam1_cam0_rect;

    std::cerr << "[Rectification] fx=" << fx_rect << " fy=" << fy_rect
              << " cx=" << cx_rect << " cy=" << cy_rect
              << " baseline=" << baseline_rect << "\n";

    vio::MetalContext* metal_context = nullptr;
    vio::MetalUndistort* metal_undistort_left = nullptr;
    vio::MetalUndistort* metal_undistort_right = nullptr;

    vio::MetalFastDetector* metal_fast = nullptr;
    vio::MetalHarrisResponse* metal_harris = nullptr;
    vio::MetalORBDescriptor* metal_orb = nullptr;
    vio::MetalStereoMatcher* metal_stereo = nullptr;
    vio::MetalKLTTracker* metal_klt = nullptr;

    bool use_metal = false;

    if (vio::MetalContext::isAvailable()) {
        metal_context = new vio::MetalContext();
        try {
            metal_undistort_left  = new vio::MetalUndistort(metal_context, map_x_left, map_y_left, 752, 480, metallib_path);
            metal_undistort_right = new vio::MetalUndistort(metal_context, map_x_right, map_y_right, 752, 480, metallib_path);

            metal_fast   = new vio::MetalFastDetector(metal_context, 752, 480, metallib_path);
            metal_harris = new vio::MetalHarrisResponse(metal_context, metallib_path);
            metal_orb    = new vio::MetalORBDescriptor(metal_context, metallib_path);

            vio::MetalStereoConfig stereo_cfg; 
            metal_stereo = new vio::MetalStereoMatcher(metal_context, metallib_path, stereo_cfg);

            metal_klt = new vio::MetalKLTTracker(metal_context, 752, 480, metallib_path);

            use_metal = metal_undistort_left->isReady()
                     && metal_fast->isReady()
                     && metal_harris->isReady()
                     && metal_orb->isReady()
                     && metal_stereo->isReady()
                     && metal_klt->isReady();

            if (use_metal)
                std::cerr << "[Metal] All GPU kernels ready\n";
            else
                std::cerr << "[Metal] Some kernels failed — falling back to CPU\n";

        } catch (const std::exception& e) {
            std::cerr << "[Metal] GPU Init Error: " << e.what() << "\n";
            use_metal = false;
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

    vio::MetalStereoCalib metal_calib;
    metal_calib.fx = (float)fx_rect;
    metal_calib.fy = (float)fy_rect;
    metal_calib.cx = (float)cx_rect;
    metal_calib.cy = (float)cy_rect;
    metal_calib.baseline = (float)baseline_rect;

    vio::VioVisualizer visualizer;

    bool pipeline_initialized = false;
    cv::Mat prev_left_undistorted;
    uint64_t frame_count = 0;
    uint64_t last_imu_ts = 0;
    int frames_since_keyframe = 0;

    vio::ImuPreintegrator preintegrator(imu_noise, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // =====================================================================
    // VIO BACKGROUND THREAD
    // =====================================================================
    std::thread vio_thread([&]() {
        @autoreleasepool {
            bool first_viz_frame = true;
            Eigen::Vector3d gt_start_pos = Eigen::Vector3d::Zero();

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

                cv::Mat left_undist, right_undist;
                if (use_metal) {
                    left_undist  = metal_undistort_left->undistort(frame.left);
                    right_undist = metal_undistort_right->undistort(frame.right);
                } else {
                    cv::remap(frame.left,  left_undist,  map_x_left,  map_y_left,  cv::INTER_LINEAR);
                    cv::remap(frame.right, right_undist, map_x_right, map_y_right, cv::INTER_LINEAR);
                }

                Eigen::Vector3d current_est_pos = Eigen::Vector3d::Zero();
                bool pose_valid = false;

                if (!pipeline_initialized) {
                    auto init_state = getInitialState(ground_truth, frame.timestamp_ns);
                    preintegrator.reset(init_state.bias_gyro, init_state.bias_accel);
                    optimizer.initialize(init_state);

                    std::vector<cv::KeyPoint> left_kpts, right_kpts;
                    cv::Mat left_desc, right_desc;
                    std::vector<vio::StereoMatcher::StereoMatch> stereo_matches;

                    if (use_metal) {
                        auto det_left  = metalDetect(*metal_fast, *metal_harris, *metal_orb,
                                                     metal_undistort_left->outputTexture(), 752, 480);
                        auto det_right = metalDetect(*metal_fast, *metal_harris, *metal_orb,
                                                     metal_undistort_right->outputTexture(), 752, 480);

                        left_kpts  = det_left.keypoints;   left_desc  = det_left.descriptors;
                        right_kpts = det_right.keypoints;  right_desc = det_right.descriptors;

                        auto metal_matches = metal_stereo->match(
                            det_left.corners, det_left.descs,
                            det_right.corners, det_right.descs, metal_calib);
                        stereo_matches = metalToVioStereoMatches(metal_matches);

                        std::cerr << "[Metal Init] FAST=" << metal_fast->lastCornerCount()
                                  << " → NMS=" << det_left.corners.size()
                                  << " stereo=" << stereo_matches.size() << "\n";
                    } else {
                        auto det = detector.detect(left_undist);
                        auto det_right_cv = detector.detect(right_undist);
                        left_kpts  = det.keypoints;          left_desc  = det.descriptors;
                        right_kpts = det_right_cv.keypoints; right_desc = det_right_cv.descriptors;
                        stereo_matches = stereo_matcher.match(left_kpts, left_desc, right_kpts, right_desc, calib);
                    }

                    feature_manager.addNewFeatures(frame.timestamp_ns, left_kpts, left_desc, stereo_matches);

                    auto world_lms = feature_manager.getInitializedLandmarksWorld(
                        init_state.position, init_state.rotation, calib.T_cam0_imu);
                    optimizer.setLandmarks(world_lms);

                    auto init_obs = feature_manager.getObservationsForFrame(frame.timestamp_ns);
                    optimizer.addObservations(frame.timestamp_ns, init_obs);

                    traj_writer.writePose(frame.timestamp_ns, init_state.position, init_state.rotation);

                    current_est_pos = init_state.position;
                    pose_valid = true;

                    prev_left_undistorted = left_undist.clone();
                    if (use_metal) {
                        metal_klt->buildPyramid(left_undist.data, (int)left_undist.step, true);
                    }
                    pipeline_initialized = true;
                    
                } else {

                    auto prev_points = feature_manager.getCurrentPoints();
                    int num_tracked = 0;
                    std::vector<cv::Point2f> tracked_points;
                    std::vector<bool> track_status;

                    if (use_metal && !prev_points.empty()) {
                        metal_klt->buildPyramid(left_undist.data, (int)left_undist.step, false);

                        std::vector<float> px(prev_points.size()), py(prev_points.size());
                        for (size_t i = 0; i < prev_points.size(); i++) {
                            px[i] = prev_points[i].x;
                            py[i] = prev_points[i].y;
                        }

                        auto klt_result = metal_klt->track(px, py);

                        tracked_points.resize(prev_points.size());
                        track_status.resize(prev_points.size(), false);
                        for (size_t i = 0; i < prev_points.size(); i++) {
                            tracked_points[i] = cv::Point2f(klt_result.tracked_x[i], klt_result.tracked_y[i]);
                            track_status[i] = klt_result.status[i];
                        }
                        num_tracked = klt_result.num_tracked;

                        if (num_tracked >= 8) {
                            std::vector<cv::Point2f> pts1, pts2;
                            std::vector<int> idx_map;
                            
                            for (size_t i = 0; i < prev_points.size(); i++) {
                                if (track_status[i]) {
                                    pts1.push_back(prev_points[i]);
                                    pts2.push_back(tracked_points[i]);
                                    idx_map.push_back((int)i);
                                }
                            }

                            if (pts1.size() >= 8) {
                                std::vector<uchar> ransac_mask;
                                cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 0.5, 0.99, ransac_mask);
                                
                                int good_tracks = 0;
                                for (size_t i = 0; i < ransac_mask.size(); i++) {
                                    if (ransac_mask[i] == 0) {
                                        track_status[idx_map[i]] = false; 
                                    } else {
                                        good_tracks++;
                                    }
                                }
                                num_tracked = good_tracks;
                            }
                        }
                    } else {
                        auto track_result = tracker.track(prev_left_undistorted, left_undist, prev_points);
                        tracked_points = track_result.tracked_points;
                        track_status = track_result.status;
                        num_tracked = track_result.num_tracked;
                    }

                    feature_manager.updateTracks(frame.timestamp_ns, feature_manager.getCurrentIds(),
                                                 tracked_points, track_status);

                    double parallax = vio::TemporalTracker::averageParallax(
                        prev_points, tracked_points, track_status, calib.intrinsics_left[0]);
                    frames_since_keyframe++;

                    if (kf_policy.shouldInsertKeyframe(num_tracked, parallax, frames_since_keyframe)) {

                        auto tracked_pts = feature_manager.getCurrentPoints();
                        auto tracked_ids = feature_manager.getCurrentIds();

                        std::vector<cv::KeyPoint> right_kpts;
                        cv::Mat right_desc;
                        std::vector<vio::CornerPoint> right_corners;
                        std::vector<vio::ORBDescriptorOutput> right_descs;

                        if (use_metal) {
                            auto det_right = metalDetect(*metal_fast, *metal_harris, *metal_orb,
                                                         metal_undistort_right->outputTexture(), 752, 480);
                            right_kpts    = det_right.keypoints;
                            right_desc    = det_right.descriptors;
                            right_corners = det_right.corners;
                            right_descs   = det_right.descs;
                        } else {
                            auto det_right_cv = detector.detect(right_undist);
                            right_kpts = det_right_cv.keypoints;
                            right_desc = det_right_cv.descriptors;
                        }

                        std::vector<vio::StereoMatcher::StereoMatch> tracked_stereo;

                        if (use_metal) {
                            std::vector<vio::CornerPoint> tracked_corners(tracked_pts.size());
                            for (size_t i = 0; i < tracked_pts.size(); i++) {
                                tracked_corners[i].position[0] = tracked_pts[i].x;
                                tracked_corners[i].position[1] = tracked_pts[i].y;
                                tracked_corners[i].response = 0;
                                tracked_corners[i].pyramid_level = 0;
                            }

                            auto tracked_orb = metal_orb->describe(
                                metal_undistort_left->outputTexture(), tracked_corners);

                            auto metal_matches = metal_stereo->match(
                                tracked_corners, tracked_orb,
                                right_corners, right_descs, metal_calib);
                            tracked_stereo = metalToVioStereoMatches(metal_matches);
                        } else {
                            std::vector<cv::KeyPoint> tracked_kpts;
                            for (size_t i = 0; i < tracked_pts.size(); i++) {
                                cv::KeyPoint kp(tracked_pts[i], 31.0f);
                                kp.class_id = static_cast<int>(i);
                                tracked_kpts.push_back(kp);
                            }

                            cv::Mat tracked_desc;
                            auto orb_tmp = cv::ORB::create();
                            orb_tmp->compute(left_undist, tracked_kpts, tracked_desc);

                            tracked_stereo = stereo_matcher.match(
                                tracked_kpts, tracked_desc,
                                right_kpts, right_desc, calib);

                            for (auto& m : tracked_stereo) {
                                m.left_idx = tracked_kpts[m.left_idx].class_id;
                            }
                        }

                        feature_manager.updateStereoForTracked(
                            frame.timestamp_ns, tracked_ids, tracked_stereo);

                        std::vector<cv::KeyPoint> new_kpts;
                        cv::Mat new_desc;
                        std::vector<vio::StereoMatcher::StereoMatch> new_stereo;

                        if (use_metal) {
                            auto existing = feature_manager.getCurrentPoints();
                            auto new_det = metalDetect(*metal_fast, *metal_harris, *metal_orb,
                                                       metal_undistort_left->outputTexture(), 752, 480,
                                                       400, &existing, 15);
                            new_kpts = new_det.keypoints;
                            new_desc = new_det.descriptors;

                            if (!new_det.corners.empty()) {
                                auto metal_matches = metal_stereo->match(
                                    new_det.corners, new_det.descs,
                                    right_corners, right_descs, metal_calib);
                                new_stereo = metalToVioStereoMatches(metal_matches);
                            }
                        } else {
                            auto new_det = detector.detect(left_undist, feature_manager.getCurrentPoints(), 15);
                            new_kpts = new_det.keypoints;
                            new_desc = new_det.descriptors;
                            new_stereo = stereo_matcher.match(new_kpts, new_desc, right_kpts, right_desc, calib);
                        }

                        feature_manager.addNewFeatures(frame.timestamp_ns, new_kpts, new_desc, new_stereo);

                        auto current_state = optimizer.latestState();
                        auto world_lms = feature_manager.getInitializedLandmarksWorld(
                            current_state.position, current_state.rotation, calib.T_cam0_imu);
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
                            Eigen::Vector3d pred_pos = prev_state.position + prev_state.velocity * dt
                                + 0.5 * g * dt * dt + R_prev * preint_result.delta_p;
                            Eigen::Quaterniond pred_rot = prev_state.rotation * preint_result.delta_R;
                            pred_rot.normalize();
                            traj_writer.writePose(frame.timestamp_ns, pred_pos, pred_rot);

                            current_est_pos = pred_pos;
                            pose_valid = true;
                        }
                    }

                    prev_left_undistorted = left_undist.clone();
                    if (use_metal) {
                        metal_klt->buildPyramid(left_undist.data, (int)left_undist.step, true);
                    }
                }
                
                if (pose_valid) {
                    auto current_gt = getInitialState(ground_truth, frame.timestamp_ns);
                    
                    if (first_viz_frame) {
                        gt_start_pos = current_gt.position;
                        first_viz_frame = false;
                    }

                    visualizer.addEstimate(current_est_pos);
                    visualizer.addGroundTruth(current_gt.position);
                }

                frame_count++;
                profiler.endFrame();
            }
            visualizer.stop(); 
        }
    });

    // =====================================================================
    // RUN VIZUALIZER (MUST BE ON MAIN THREAD)
    // =====================================================================
    visualizer.run();

    if (vio_thread.joinable()) {
        vio_thread.join();
    }

    delete metal_klt;
    delete metal_stereo;
    delete metal_orb;
    delete metal_harris;
    delete metal_fast;
    delete metal_undistort_left;
    delete metal_undistort_right;
    delete metal_context;

    }
    return 0;
}