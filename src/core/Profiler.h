#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <fstream>

namespace vio {

struct FrameTiming {
    uint64_t frame_id = 0;
    uint64_t timestamp_ns = 0;
    double load_ms = 0.0;
    double undistort_ms = 0.0;
    double detect_ms = 0.0;
    double describe_ms = 0.0;
    double stereo_match_ms = 0.0;
    double stereo_retrack_ms = 0.0;
    double temporal_track_ms = 0.0;
    double preintegration_ms = 0.0;
    double optimize_ms = 0.0;
    double total_ms = 0.0;
    int num_features_detected = 0;
    int num_stereo_matches = 0;
    int num_stereo_retracked = 0;
    int num_tracked = 0;
    int num_landmarks_initialized = 0;
    int num_landmarks_in_window = 0;
};

class Profiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    void beginFrame(uint64_t frame_id, uint64_t timestamp_ns);
    void startStage(const std::string& name);
    double endStage(const std::string& name); // returns elapsed ms

    void setCount(const std::string& name, int value);

    void endFrame();

    const FrameTiming& currentFrame() const { return current_; }
    const std::vector<FrameTiming>& history() const { return history_; }

    void writeCSV(const std::string& path) const;

    void printSummary() const;

private:
    FrameTiming current_;
    std::vector<FrameTiming> history_;
    std::unordered_map<std::string, TimePoint> stage_starts_;
    TimePoint frame_start_;
};

}
