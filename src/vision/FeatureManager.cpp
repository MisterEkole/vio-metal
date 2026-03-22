#include "FeatureManager.h"

namespace vio {

void FeatureManager::addNewFeatures(
    uint64_t timestamp,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors,
    const std::vector<StereoMatcher::StereoMatch>& stereo_matches)
{
    std::unordered_map<int, const StereoMatcher::StereoMatch*> stereo_lookup;
    for (const auto& m : stereo_matches) {
        stereo_lookup[m.left_idx] = &m;
    }

    for (int i = 0; i < (int)keypoints.size(); i++) {
        uint64_t fid = next_feature_id_++;
        FeatureTrack track;
        track.feature_id = fid;
        track.frame_timestamps.push_back(timestamp);
        track.observations_left.emplace_back(keypoints[i].pt.x, keypoints[i].pt.y);

        auto it = stereo_lookup.find(i);
        if (it != stereo_lookup.end()) {
            const auto& sm = *(it->second);
            track.observations_right.emplace_back(
                keypoints[i].pt.x - sm.disparity, keypoints[i].pt.y);
            track.landmark_3d = sm.point_3d;
            track.landmark_initialized = true;
        } else {
            track.observations_right.emplace_back(-1.0, -1.0);
            track.landmark_initialized = false;
        }

        tracks_[fid] = std::move(track);
        current_ids_.push_back(fid);
        current_points_.push_back(keypoints[i].pt);
    }
}

void FeatureManager::updateTracks(
    uint64_t timestamp,
    const std::vector<uint64_t>& tracked_ids,
    const std::vector<cv::Point2f>& tracked_points,
    const std::vector<bool>& status)
{
    std::vector<uint64_t> new_ids;
    std::vector<cv::Point2f> new_points;

    for (size_t i = 0; i < tracked_ids.size(); i++) {
        if (!status[i]) continue;

        auto it = tracks_.find(tracked_ids[i]);
        if (it == tracks_.end()) continue;

        auto& track = it->second;
        track.frame_timestamps.push_back(timestamp);
        track.observations_left.emplace_back(tracked_points[i].x, tracked_points[i].y);
        track.observations_right.emplace_back(-1.0, -1.0);

        new_ids.push_back(tracked_ids[i]);
        new_points.push_back(tracked_points[i]);
    }

    current_ids_ = std::move(new_ids);
    current_points_ = std::move(new_points);
}

void FeatureManager::updateStereoForTracked(
    uint64_t timestamp,
    const std::vector<uint64_t>& tracked_ids,
    const std::vector<StereoMatcher::StereoMatch>& stereo_matches)
{
    for (const auto& m : stereo_matches) {
        if (m.left_idx < 0 || m.left_idx >= static_cast<int>(tracked_ids.size())) continue;
        
        uint64_t fid = tracked_ids[m.left_idx];
        auto it = tracks_.find(fid);
        if (it == tracks_.end()) continue;

        auto& track = it->second;
        
        if (!track.observations_right.empty() && !track.observations_left.empty()) {
            double u_left = track.observations_left.back().x();
            double v_left = track.observations_left.back().y();
            track.observations_right.back() = Eigen::Vector2d(
                u_left - m.disparity, v_left);
        }
        
        if (!track.landmark_initialized) {
            track.landmark_3d = m.point_3d;
            track.landmark_initialized = true;
        }
    }
}

std::unordered_map<uint64_t, Eigen::Vector3d> FeatureManager::getInitializedLandmarksWorld(
    const Eigen::Vector3d& p_wb, 
    const Eigen::Quaterniond& q_wb, 
    const Eigen::Matrix4d& T_cam_imu) const 
{
    std::unordered_map<uint64_t, Eigen::Vector3d> landmarks_world;
    
    // T_cam_imu: IMU→camera. Invert to get camera→body.
    Eigen::Matrix3d R_ci = T_cam_imu.block<3,3>(0,0);
    Eigen::Vector3d t_ci = T_cam_imu.block<3,1>(0,3);
    Eigen::Matrix3d R_ic = R_ci.transpose();
    Eigen::Vector3d t_ic = -R_ci.transpose() * t_ci;

    for (const auto& [fid, track] : tracks_) {
        if (track.landmark_initialized) {
            Eigen::Vector3d P_body = R_ic * track.landmark_3d + t_ic;
            landmarks_world[fid] = q_wb * P_body + p_wb;
        }
    }
    return landmarks_world;
}

std::vector<cv::Point2f> FeatureManager::getCurrentPoints() const { return current_points_; }
std::vector<uint64_t> FeatureManager::getCurrentIds() const { return current_ids_; }

std::vector<FeatureObservation> FeatureManager::getObservationsForFrame(uint64_t timestamp) const {
    std::vector<FeatureObservation> obs;
    for (const auto& [fid, track] : tracks_) {
        for (size_t i = 0; i < track.frame_timestamps.size(); i++) {
            if (track.frame_timestamps[i] == timestamp) {
                FeatureObservation o;
                o.feature_id = fid;
                o.pixel_left = track.observations_left[i];
                o.pixel_right = track.observations_right[i];
                o.has_stereo = (o.pixel_right.x() >= 0);
                o.landmark_3d = track.landmark_3d;
                o.landmark_initialized = track.landmark_initialized;
                obs.push_back(o);
                break;
            }
        }
    }
    return obs;
}

std::vector<const FeatureTrack*> FeatureManager::getActiveTracksInWindow(
    const std::vector<uint64_t>& window_timestamps) const {
    std::vector<const FeatureTrack*> result;
    for (const auto& [fid, track] : tracks_) {
        if (!track.landmark_initialized) continue;
        int count = 0;
        for (const auto& ts : track.frame_timestamps) {
            for (const auto& wts : window_timestamps) {
                if (ts == wts) { count++; break; }
            }
        }
        if (count >= 2) result.push_back(&track);
    }
    return result;
}

void FeatureManager::pruneDeadTracks(int min_track_length) {
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        bool is_current = std::find(current_ids_.begin(), current_ids_.end(), it->first) != current_ids_.end();
        if (!is_current && (int)it->second.frame_timestamps.size() < min_track_length) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

void FeatureManager::removeTracksOlderThan(uint64_t timestamp) {
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second.frame_timestamps.back() < timestamp) {
            bool is_current = std::find(current_ids_.begin(), current_ids_.end(), it->first) != current_ids_.end();
            if (!is_current) {
                it = tracks_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

}