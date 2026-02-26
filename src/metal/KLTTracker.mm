#import "KLTTracker.h"
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace vio {

struct KLTParamsGPU {
    uint32_t n_points;
    uint32_t max_iterations;
    float    epsilon;
    int32_t  win_radius;
    uint32_t max_level;
    float    min_eigenvalue;
};
MetalKLTTracker::~MetalKLTTracker() {
}

MetalKLTTracker::MetalKLTTracker(MetalContext* context, int width, int height,
                                 const std::string& metallib_path, const KLTConfig& config)
    : context_(context), width_(width), height_(height), config_(config), last_n_points_(0)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) return;
    
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;
    klt_pipeline_ = (__bridge KLTPipelinePtr)context_->getPipeline("klt_track_forward", (__bridge void*)library);

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    int w = width_, h = height_;
    for (int i = 0; i < 4; i++) {
        prev_pyramid_[i] = createLevelTexture(w, h);
        curr_pyramid_[i] = createLevelTexture(w, h);
        w = (w + 1) / 2; h = (h + 1) / 2;
    }

    prev_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2 options:MTLResourceStorageModeShared];
    curr_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2 options:MTLResourceStorageModeShared];
    back_pts_buffer_ = [device newBufferWithLength:max_points_ * sizeof(float) * 2 options:MTLResourceStorageModeShared];
    status_buffer_   = [device newBufferWithLength:max_points_ * sizeof(uint8_t) options:MTLResourceStorageModeShared];
    back_status_buffer_ = [device newBufferWithLength:max_points_ * sizeof(uint8_t) options:MTLResourceStorageModeShared];
    params_buffer_   = [device newBufferWithLength:sizeof(KLTParamsGPU) options:MTLResourceStorageModeShared];

    ready_ = (klt_pipeline_ != nil && prev_pyramid_[0] != nil);
}

KLTTexturePtr MetalKLTTracker::createLevelTexture(int w, int h) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:w height:h mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    return [device newTextureWithDescriptor:desc];
}

void MetalKLTTracker::buildPyramid(const uint8_t* image_data, int stride, bool is_previous) {
    if (!ready_) return;
    KLTTexturePtr __strong * pyr = is_previous ? prev_pyramid_ : curr_pyramid_;

    [pyr[0] replaceRegion:MTLRegionMake2D(0, 0, width_, height_) mipmapLevel:0 withBytes:image_data bytesPerRow:stride];

    int prev_w = width_, prev_h = height_, prev_stride = stride;
    std::vector<uint8_t> prev_level(image_data, image_data + stride * height_);

    for (int level = 1; level <= (int)config_.max_level; level++) {
        int cur_w = (prev_w + 1) / 2; int cur_h = (prev_h + 1) / 2;
        std::vector<uint8_t> cur_level(cur_w * cur_h);
        for (int y = 0; y < cur_h; y++) {
            for (int x = 0; x < cur_w; x++) {
                int sum = (int)prev_level[(y*2)*prev_stride + (x*2)] + (int)prev_level[(y*2)*prev_stride + std::min(x*2+1, prev_w-1)] +
                          (int)prev_level[std::min(y*2+1, prev_h-1)*prev_stride + (x*2)] + (int)prev_level[std::min(y*2+1, prev_h-1)*prev_stride + std::min(x*2+1, prev_w-1)];
                cur_level[y * cur_w + x] = (uint8_t)((sum + 2) / 4);
            }
        }
        [pyr[level] replaceRegion:MTLRegionMake2D(0, 0, cur_w, cur_h) mipmapLevel:0 withBytes:cur_level.data() bytesPerRow:cur_w];
        prev_level = std::move(cur_level);
        prev_stride = cur_w; prev_w = cur_w; prev_h = cur_h;
    }
}

void MetalKLTTracker::dispatchKLT(KLTTexturePtr const * from_pyr, KLTTexturePtr const  * to_pyr,
                                  BufferPtr in_pts, BufferPtr out_pts, BufferPtr status_buf,
                                  uint32_t n_points, void* cmdBufPtr)
{
    id<MTLCommandBuffer> cmdBuf = (__bridge id<MTLCommandBuffer>)cmdBufPtr;
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:klt_pipeline_];

    for (int i = 0; i < 4; i++) {
        [enc setTexture:from_pyr[i] atIndex:i];
        [enc setTexture:to_pyr[i] atIndex:i + 4];
    }

    [enc setBuffer:in_pts offset:0 atIndex:0];
    [enc setBuffer:out_pts offset:0 atIndex:1];
    [enc setBuffer:status_buf offset:0 atIndex:2];
    [enc setBuffer:params_buffer_ offset:0 atIndex:3];

    [enc dispatchThreads:MTLSizeMake(n_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(n_points, 256u), 1, 1)];
    [enc endEncoding];
}

void MetalKLTTracker::encodeTrack(const std::vector<cv::Point2f>& points) {

    if (!ready_ || points.empty()) return;
    
    // Determine point count (clamped to GPU buffer size)
    uint32_t n = (uint32_t)std::min(points.size(), (size_t)max_points_);
    last_n_points_ = n;

    // Copy points to the Metal Shared Buffer and update internal cache
    float* pts_gpu = (float*)[prev_pts_buffer_ contents];
    cached_px_.resize(n);
    cached_py_.resize(n);

    for (uint32_t i = 0; i < n; i++) { 
        // Fill GPU buffer
        pts_gpu[i*2]   = points[i].x; 
        pts_gpu[i*2+1] = points[i].y; 
        
        // Fill CPU cache for FB-error validation later
        cached_px_[i] = points[i].x;
        cached_py_[i] = points[i].y;
    }

    KLTParamsGPU params = {
        n, 
        (uint32_t)config_.max_iterations, 
        config_.epsilon, 
        config_.win_radius, 
        (uint32_t)config_.max_level, 
        config_.min_eigenvalue
    };
    memcpy([params_buffer_ contents], &params, sizeof(KLTParamsGPU));
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    
    // // Fix "passing address of non-local object to __autoreleasing" by using __strong pointers
    // KLTTexturePtr __strong * p_pyr = prev_pyramid_;
    // KLTTexturePtr __strong * c_pyr = curr_pyramid_;
    // dispatchKLT(p_pyr, c_pyr, prev_pts_buffer_, curr_pts_buffer_, status_buffer_, n, (__bridge void*)cmdBuf);

    KLTTexturePtr const * p_pyr = prev_pyramid_;
    KLTTexturePtr const * c_pyr = curr_pyramid_;

    dispatchKLT(p_pyr, c_pyr, prev_pts_buffer_, curr_pts_buffer_, status_buffer_, n, (__bridge void*)cmdBuf);
   
    dispatchKLT(c_pyr, p_pyr, curr_pts_buffer_, back_pts_buffer_, back_status_buffer_, n, (__bridge void*)cmdBuf);


    [cmdBuf commit];
    context_->setLastBuffer((__bridge void*)cmdBuf);
}

KLTResult MetalKLTTracker::getResults() {
    KLTResult result;
    if (last_n_points_ == 0) return result;

    float* fwd_pts = (float*)[curr_pts_buffer_ contents];
    float* back_pts = (float*)[back_pts_buffer_ contents];
    uint8_t* f_status = (uint8_t*)[status_buffer_ contents];
    uint8_t* b_status = (uint8_t*)[back_status_buffer_ contents];

    result.tracked_x.resize(last_n_points_); result.tracked_y.resize(last_n_points_);
    result.status.resize(last_n_points_, false); result.num_tracked = 0;

    float fb_thresh_sq = config_.fb_threshold * config_.fb_threshold;
    for (uint32_t i = 0; i < last_n_points_; i++) {
        if (!f_status[i] || !b_status[i]) continue;
        float dx = cached_px_[i] - back_pts[i*2], dy = cached_py_[i] - back_pts[i*2+1];
        if (dx*dx + dy*dy > fb_thresh_sq) continue;
        result.tracked_x[i] = fwd_pts[i*2]; result.tracked_y[i] = fwd_pts[i*2+1];
        result.status[i] = true; result.num_tracked++;
    }
    return result;
}

} // namespace vio