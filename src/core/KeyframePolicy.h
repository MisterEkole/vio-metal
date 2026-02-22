#pragma once

namespace vio {

class KeyframePolicy {
public:
    struct Config {
        int min_tracked_features = 80;
        double min_parallax_deg = 1.0;
        int min_frames_between = 2;
        int max_frames_between = 10;
    };
    KeyframePolicy();

    explicit KeyframePolicy(const Config& config);

    bool shouldInsertKeyframe(
        int num_tracked_features,
        double average_parallax_deg,
        int frames_since_last_keyframe
    ) const;

    const Config& config() const { return config_; }

private:
    Config config_;
};

}
