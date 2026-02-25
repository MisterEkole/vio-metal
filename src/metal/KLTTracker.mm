#import "KLTTracker.h"
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <iostream>
#include <cstring>
#include <cmath>

namespace vio {

// Must match KLTTracker.metal
struct KLTParamsGPU {
    uint32_t n_points;
    uint32_t max_iterations;
    float    epsilon;
    int32_t  win_radius;
    uint32_t max_level;
    float    min_eigenvalue;
};

MetalKLTTracker::MetalKLTTracker(MetalContext* context,
                                 int width, int height,
                                 const std::string& metallib_path,
                                 const KLTConfig& config)
    : context_(context), width_(width), height_(height), config_(config)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalKLT] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;

    void* pipe_ptr = context_->getPipeline("klt_track_forward", (__bridge void*)library);
    klt_pipeline_ = (__bridge id<MTLComputePipelineState>)pipe_ptr;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    // Create pyramid textures for both frames (4 levels each)
    int w = width_, h = height_;
    for (int i = 0; i < 4; i++) {
        prev_pyramid_[i] = createLevelTexture(w, h);
        curr_pyramid_[i] = createLevelTexture(w, h);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    // Point buffers
    prev_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2
                                           options:MTLResourceStorageModeShared];
    curr_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2
                                           options:MTLResourceStorageModeShared];
    back_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2
                                           options:MTLResourceStorageModeShared];
    status_buffer_ = [device newBufferWithLength:max_points_ * sizeof(uint8_t)
                                         options:MTLResourceStorageModeShared];
    back_status_buffer_ = [device newBufferWithLength:max_points_ * sizeof(uint8_t)
                                              options:MTLResourceStorageModeShared];
    params_buffer_ = [device newBufferWithLength:sizeof(KLTParamsGPU)
                                         options:MTLResourceStorageModeShared];

    ready_ = (klt_pipeline_ != nil && prev_pyramid_[0] != nil);
    if (ready_) {
        std::cerr << "[MetalKLT] Ready — win=" << (2*config_.win_radius+1)
                  << "x" << (2*config_.win_radius+1)
                  << " levels=" << (config_.max_level+1)
                  << " fb_thresh=" << config_.fb_threshold << "\n";
    }
}

TexturePtr MetalKLTTracker::createLevelTexture(int w, int h) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                          width:w
                                                         height:h
                                                        mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    return [device newTextureWithDescriptor:desc];
}

void MetalKLTTracker::buildPyramid(const uint8_t* image_data, int stride, bool is_previous) {
    if (!ready_) return;

    __strong TexturePtr* pyr = is_previous ? prev_pyramid_ : curr_pyramid_;

    // Level 0: upload the full-resolution image
    MTLRegion region = MTLRegionMake2D(0, 0, width_, height_);
    [pyr[0] replaceRegion:region mipmapLevel:0
                withBytes:image_data
              bytesPerRow:stride];

    // Levels 1-3: simple 2×2 box downsample on CPU, then upload
    // (Swap to MPS pyramid later for production)
    int prev_w = width_, prev_h = height_;
    std::vector<uint8_t> prev_level(image_data, image_data + stride * height_);
    int prev_stride = stride;

    for (int level = 1; level <= config_.max_level; level++) {
        int cur_w = (prev_w + 1) / 2;
        int cur_h = (prev_h + 1) / 2;
        std::vector<uint8_t> cur_level(cur_w * cur_h);

        for (int y = 0; y < cur_h; y++) {
            for (int x = 0; x < cur_w; x++) {
                int sx = x * 2, sy = y * 2;
                // Average 2×2 block (clamp to image bounds)
                int sx1 = std::min(sx + 1, prev_w - 1);
                int sy1 = std::min(sy + 1, prev_h - 1);
                int sum = (int)prev_level[sy * prev_stride + sx]
                        + (int)prev_level[sy * prev_stride + sx1]
                        + (int)prev_level[sy1 * prev_stride + sx]
                        + (int)prev_level[sy1 * prev_stride + sx1];
                cur_level[y * cur_w + x] = (uint8_t)((sum + 2) / 4);
            }
        }

        MTLRegion lvl_region = MTLRegionMake2D(0, 0, cur_w, cur_h);
        [pyr[level] replaceRegion:lvl_region mipmapLevel:0
                        withBytes:cur_level.data()
                      bytesPerRow:cur_w];

        prev_level = std::move(cur_level);
        prev_stride = cur_w;
        prev_w = cur_w;
        prev_h = cur_h;
    }
}

void MetalKLTTracker::dispatchKLT(__strong TexturePtr* from_pyr, __strong TexturePtr* to_pyr,
                                  BufferPtr in_pts, BufferPtr out_pts,
                                  BufferPtr status_buf, uint32_t n_points,
                                  id<MTLCommandBuffer> cmdBuf)
{
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:klt_pipeline_];

    // Previous pyramid: textures 0-3
    [enc setTexture:from_pyr[0] atIndex:0];
    [enc setTexture:from_pyr[1] atIndex:1];
    [enc setTexture:from_pyr[2] atIndex:2];
    [enc setTexture:from_pyr[3] atIndex:3];

    // Current pyramid: textures 4-7
    [enc setTexture:to_pyr[0] atIndex:4];
    [enc setTexture:to_pyr[1] atIndex:5];
    [enc setTexture:to_pyr[2] atIndex:6];
    [enc setTexture:to_pyr[3] atIndex:7];

    [enc setBuffer:in_pts     offset:0 atIndex:0];
    [enc setBuffer:out_pts    offset:0 atIndex:1];
    [enc setBuffer:status_buf offset:0 atIndex:2];
    [enc setBuffer:params_buffer_ offset:0 atIndex:3];

    MTLSize grid  = MTLSizeMake(n_points, 1, 1);
    MTLSize group = MTLSizeMake(std::min(n_points, 256u), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
}

KLTResult MetalKLTTracker::track(const std::vector<float>& prev_x,
                                 const std::vector<float>& prev_y)
{
    KLTResult result;
    if (!ready_ || prev_x.empty()) return result;

    uint32_t n = (uint32_t)std::min(prev_x.size(), (size_t)max_points_);

    // Pack (x,y) pairs into float2 buffer
    float* pts = (float*)[prev_pts_buffer_ contents];
    for (uint32_t i = 0; i < n; i++) {
        pts[i * 2]     = prev_x[i];
        pts[i * 2 + 1] = prev_y[i];
    }

    // Set params
    KLTParamsGPU params;
    params.n_points       = n;
    params.max_iterations = config_.max_iterations;
    params.epsilon        = config_.epsilon;
    params.win_radius     = config_.win_radius;
    params.max_level      = config_.max_level;
    params.min_eigenvalue = config_.min_eigenvalue;
    memcpy([params_buffer_ contents], &params, sizeof(KLTParamsGPU));

    // === Forward pass: prev → curr ===
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    dispatchKLT(prev_pyramid_, curr_pyramid_,
                prev_pts_buffer_, curr_pts_buffer_, status_buffer_,
                n, cmdBuf);

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    double fwd_time = ([cmdBuf GPUEndTime] - [cmdBuf GPUStartTime]) * 1000.0;

    // === Backward pass: curr → prev (for forward-backward validation) ===
    id<MTLCommandBuffer> cmdBuf2 = [queue commandBuffer];

    dispatchKLT(curr_pyramid_, prev_pyramid_,
                curr_pts_buffer_, back_pts_buffer_, back_status_buffer_,
                n, cmdBuf2);

    [cmdBuf2 commit];
    [cmdBuf2 waitUntilCompleted];

    double bwd_time = ([cmdBuf2 GPUEndTime] - [cmdBuf2 GPUStartTime]) * 1000.0;
    last_gpu_ms_ = fwd_time + bwd_time;

    // === Read results and validate ===
    float*   fwd_pts    = (float*)[curr_pts_buffer_ contents];
    float*   back_pts   = (float*)[back_pts_buffer_ contents];
    uint8_t* fwd_status = (uint8_t*)[status_buffer_ contents];
    uint8_t* bwd_status = (uint8_t*)[back_status_buffer_ contents];

    result.tracked_x.resize(n);
    result.tracked_y.resize(n);
    result.status.resize(n, false);
    result.num_tracked = 0;

    float fb_thresh_sq = config_.fb_threshold * config_.fb_threshold;

    for (uint32_t i = 0; i < n; i++) {
        if (!fwd_status[i] || !bwd_status[i]) continue;

        float tx = fwd_pts[i * 2];
        float ty = fwd_pts[i * 2 + 1];

        // Forward-backward consistency: does back-tracking land near the original?
        float bx = back_pts[i * 2];
        float by = back_pts[i * 2 + 1];
        float dx = prev_x[i] - bx;
        float dy = prev_y[i] - by;

        if (dx * dx + dy * dy > fb_thresh_sq) continue;

        result.tracked_x[i] = tx;
        result.tracked_y[i] = ty;
        result.status[i] = true;
        result.num_tracked++;
    }

    return result;
}

}