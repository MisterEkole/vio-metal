#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace vio { class MetalContext; }

#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLComputePipelineState> PipelinePtr;
typedef id<MTLBuffer>              BufferPtr;
typedef id<MTLTexture>             TexturePtr;
#else
typedef void* PipelinePtr;
typedef void* BufferPtr;
typedef void* TexturePtr;
#endif

namespace vio {

struct KLTConfig {
    int   max_iterations = 30;
    float epsilon        = 0.01f;   // convergence threshold (pixels)
    int   win_radius     = 10;      // 10 → 21×21 window
    int   max_level      = 3;       // pyramid levels 0-3
    float min_eigenvalue = 1e-4f;   // reject textureless patches
    float fb_threshold   = 1.0f;    // forward-backward consistency (pixels)
};

struct KLTResult {
    std::vector<float> tracked_x;     // output x-coordinates
    std::vector<float> tracked_y;     // output y-coordinates
    std::vector<bool>  status;        // true = successfully tracked
    int num_tracked = 0;
};

class MetalKLTTracker {
public:
    MetalKLTTracker(MetalContext* context,
                    int width, int height,
                    const std::string& metallib_path,
                    const KLTConfig& config = KLTConfig());

    ~MetalKLTTracker() = default;

    /// Build image pyramid for a frame (call once per frame)
    /// Returns an opaque handle to the pyramid textures
    void buildPyramid(const uint8_t* image_data, int stride, bool is_previous);

    /// Track points from previous frame to current frame
    /// with forward-backward validation
    KLTResult track(const std::vector<float>& prev_x,
                    const std::vector<float>& prev_y);

    double lastGpuTimeMs() const { return last_gpu_ms_; }
    bool   isReady()       const { return ready_; }

private:
    MetalContext* context_;

    PipelinePtr klt_pipeline_;

    // Two sets of pyramid textures: previous and current frame
    TexturePtr prev_pyramid_[4];
    TexturePtr curr_pyramid_[4];

    // Point buffers
    BufferPtr prev_pts_buffer_;
    BufferPtr curr_pts_buffer_;
    BufferPtr back_pts_buffer_;    // for forward-backward validation
    BufferPtr status_buffer_;
    BufferPtr back_status_buffer_;
    BufferPtr params_buffer_;

    int width_, height_;
    KLTConfig config_;
    uint32_t max_points_ = 1000;
    double   last_gpu_ms_ = 0.0;
    bool     ready_ = false;

    /// Dispatch the KLT kernel in one direction
    void dispatchKLT(__strong TexturePtr* from_pyr, __strong TexturePtr* to_pyr,
                     BufferPtr in_pts, BufferPtr out_pts,
                     BufferPtr status_buf, uint32_t n_points,
                     id<MTLCommandBuffer> cmdBuf);

    /// Create a single pyramid level texture
    TexturePtr createLevelTexture(int w, int h);
};

}
