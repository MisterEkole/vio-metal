#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace vio {

class MetalContext;

struct KLTConfig {
    float fb_threshold = 1.0f;
    int max_iterations = 20;
    float epsilon = 0.01f;
    int win_radius = 15;
    int max_level = 3;
    float min_eigenvalue = 0.001f;
};

struct KLTResult {
    std::vector<float> tracked_x;
    std::vector<float> tracked_y;
    std::vector<bool> status;
    int num_tracked = 0;
};

class MetalKLTTracker {
public:
    MetalKLTTracker(MetalContext* context, int width, int height, 
                    const std::string& metallib_path, const KLTConfig& config);
    ~MetalKLTTracker();

    void buildPyramid(const uint8_t* image_data, int stride, bool is_previous);
    void encodeTrack(const std::vector<float>& prev_x, const std::vector<float>& prev_y);
    KLTResult getResults();
    void swapPyramids();

private:
    // Internal helper for GPU dispatch (Implementation in .mm)
    void dispatchKLT(void* cmdBufPtr);

    MetalContext* context_;
    
    // We store these as void* in the header to remain C++ compatible
    void* klt_pipeline_;
    void* prev_pyramid_[4];
    void* curr_pyramid_[4];
    
    void* prev_pts_buffer_;
    void* curr_pts_buffer_;
    void* back_pts_buffer_;
    void* status_buffer_;
    void* back_status_buffer_;
    void* params_buffer_;

    int width_, height_;
    KLTConfig config_;
    uint32_t last_n_points_;
    const uint32_t max_points_ = 1000;
    bool ready_ = false;

    std::vector<float> cached_px_, cached_py_;
};

} // namespace vio