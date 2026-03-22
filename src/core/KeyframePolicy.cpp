#include "KeyframePolicy.h"

namespace vio {
KeyframePolicy::KeyframePolicy() : KeyframePolicy(Config{}) {}

KeyframePolicy::KeyframePolicy(const Config& config) : config_(config) {}

bool KeyframePolicy::shouldInsertKeyframe(
    int num_tracked_features,
    double average_parallax_deg,
    int frames_since_last_keyframe) const
{
    if (frames_since_last_keyframe < config_.min_frames_between) {
        return false;
    }

    if (frames_since_last_keyframe >= config_.max_frames_between) {
        return true;
    }

    if (num_tracked_features < config_.min_tracked_features) {
        return true;
    }

    if (average_parallax_deg > config_.min_parallax_deg) {
        return true;
    }

    return false;
}

} 
