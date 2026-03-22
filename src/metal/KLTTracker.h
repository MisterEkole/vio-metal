#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>

#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLComputePipelineState> KLTPipelinePtr;
typedef id<MTLTexture> KLTTexturePtr;
typedef id<MTLBuffer> BufferPtr;
#else
typedef void* KLTPipelinePtr;
typedef void* KLTTexturePtr;
typedef void* BufferPtr;
#endif

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
                    const std::string& metallib_path, const KLTConfig& config = KLTConfig());
    ~MetalKLTTracker();

    void buildPyramid(const uint8_t* image_data, int stride, bool is_previous);
    
    void encodeTrack(const std::vector<cv::Point2f>& points);
   
    
    KLTResult getResults();
    void swapPyramids();

private:
    void dispatchKLT(KLTTexturePtr const* from_pyr, KLTTexturePtr const* to_pyr,
                    BufferPtr in_pts, BufferPtr out_pts, BufferPtr status_buf,
                    uint32_t n, void* cmdBufPtr);
    
    KLTTexturePtr createLevelTexture(int w, int h);

    MetalContext* context_;
    KLTPipelinePtr klt_pipeline_;
    KLTTexturePtr prev_pyramid_[4];
    KLTTexturePtr curr_pyramid_[4];
    
    BufferPtr prev_pts_buffer_;
    BufferPtr curr_pts_buffer_;
    BufferPtr back_pts_buffer_;
    BufferPtr status_buffer_;
    BufferPtr back_status_buffer_;
    BufferPtr params_buffer_;

    bool ready_ = false;
    std::vector<float> cached_px_, cached_py_;
    int width_, height_;
    KLTConfig config_;
    uint32_t last_n_points_ = 0;
    const uint32_t max_points_ = 1000;
};

} // namespace vio