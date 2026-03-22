#include "Profiler.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace vio {

void Profiler::beginFrame(uint64_t frame_id, uint64_t timestamp_ns) {
    current_ = FrameTiming{};
    current_.frame_id = frame_id;
    current_.timestamp_ns = timestamp_ns;
    stage_starts_.clear();
    frame_start_ = Clock::now();
}

void Profiler::startStage(const std::string& name) {
    stage_starts_[name] = Clock::now();
}

double Profiler::endStage(const std::string& name) {
    auto it = stage_starts_.find(name);
    if (it == stage_starts_.end()) return 0.0;

    double ms = std::chrono::duration<double, std::milli>(Clock::now() - it->second).count();

    if (name == "load") current_.load_ms = ms;
    else if (name == "undistort") current_.undistort_ms = ms;
    else if (name == "detect") current_.detect_ms = ms;
    else if (name == "describe") current_.describe_ms = ms;
    else if (name == "stereo_match") current_.stereo_match_ms = ms;
    else if (name == "stereo_retrack") current_.stereo_retrack_ms = ms;
    else if (name == "temporal_track") current_.temporal_track_ms = ms;
    else if (name == "preintegration") current_.preintegration_ms = ms;
    else if (name == "optimize") current_.optimize_ms = ms;

    return ms;
}

void Profiler::setCount(const std::string& name, int value) {
    if (name == "features_detected") current_.num_features_detected = value;
    else if (name == "stereo_matches") current_.num_stereo_matches = value;
    else if (name == "stereo_retracked") current_.num_stereo_retracked = value;
    else if (name == "tracked") current_.num_tracked = value;
    else if (name == "landmarks_initialized") current_.num_landmarks_initialized = value;
    else if (name == "landmarks") current_.num_landmarks_in_window = value;
}

void Profiler::endFrame() {
    current_.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - frame_start_).count();
    history_.push_back(current_);
}

void Profiler::writeCSV(const std::string& path) const {
    std::ofstream f(path);
    f << "frame_id,timestamp_ns,load_ms,undistort_ms,detect_ms,describe_ms,"
      << "stereo_ms,stereo_retrack_ms,track_ms,preint_ms,optimize_ms,total_ms,"
      << "n_features,n_stereo,n_stereo_retracked,n_tracked,n_landmarks_init,n_landmarks\n";

    for (const auto& t : history_) {
        f << t.frame_id << ","
          << t.timestamp_ns << ","
          << std::fixed << std::setprecision(3)
          << t.load_ms << ","
          << t.undistort_ms << ","
          << t.detect_ms << ","
          << t.describe_ms << ","
          << t.stereo_match_ms << ","
          << t.stereo_retrack_ms << ","
          << t.temporal_track_ms << ","
          << t.preintegration_ms << ","
          << t.optimize_ms << ","
          << t.total_ms << ","
          << t.num_features_detected << ","
          << t.num_stereo_matches << ","
          << t.num_stereo_retracked << ","
          << t.num_tracked << ","
          << t.num_landmarks_initialized << ","
          << t.num_landmarks_in_window << "\n";
    }
}

void Profiler::printSummary() const {
    if (history_.empty()) return;

    auto avg = [&](auto field) {
        double sum = 0;
        for (const auto& t : history_) sum += t.*field;
        return sum / history_.size();
    };

    auto max_val = [&](auto field) {
        double mx = 0;
        for (const auto& t : history_) mx = std::max(mx, t.*field);
        return mx;
    };

    std::cout << "\n===== Profiling Summary (" << history_.size() << " frames) =====\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Stage           | Avg (ms) | Max (ms)\n";
    std::cout << "----------------|----------|--------\n";

    auto row = [&](const char* name, double FrameTiming::* f) {
        std::cout << std::left << std::setw(16) << name
                  << "| " << std::setw(8) << avg(f)
                  << " | " << max_val(f) << "\n";
    };

    row("Load",         &FrameTiming::load_ms);
    row("Undistort",    &FrameTiming::undistort_ms);
    row("Detect",       &FrameTiming::detect_ms);
    row("Describe",     &FrameTiming::describe_ms);
    row("Stereo Match", &FrameTiming::stereo_match_ms);
    row("Stereo Retrk", &FrameTiming::stereo_retrack_ms);
    row("Track",        &FrameTiming::temporal_track_ms);
    row("Preintegrate", &FrameTiming::preintegration_ms);
    row("Optimize",     &FrameTiming::optimize_ms);
    row("TOTAL",        &FrameTiming::total_ms);
    std::cout << "================================================\n\n";
}

} 
