#import "StereoMatcher.h"
#import "FastDetect.h"     // CornerPoint
#import "ORBDescriptor.h"    // ORBDescriptorOutput
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <iostream>

namespace vio {

// Must match StereoMatch.metal exactly
struct StereoParamsGPU {
    uint32_t n_left;
    uint32_t n_right;
    float    max_epipolar;
    float    min_disparity;
    float    max_disparity;
    uint32_t max_hamming;
    float    ratio_thresh;
    float    fx, fy, cx, cy, baseline;
};

MetalStereoMatcher::MetalStereoMatcher(MetalContext* context,
                                       const std::string& metallib_path,
                                       const MetalStereoConfig& config)
    : context_(context), config_(config)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalStereo] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;

    void* pipe1 = context_->getPipeline("stereo_hamming", (__bridge void*)library);
    void* pipe2 = context_->getPipeline("stereo_extract", (__bridge void*)library);
    hamming_pipeline_ = (__bridge id<MTLComputePipelineState>)pipe1;
    extract_pipeline_ = (__bridge id<MTLComputePipelineState>)pipe2;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    uint32_t N = max_keypoints_;

    // Descriptor buffers: each descriptor is 8 × uint32 = 32 bytes
    left_desc_buffer_  = [device newBufferWithLength:N * 8 * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    right_desc_buffer_ = [device newBufferWithLength:N * 8 * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

    // Keypoint buffers
    left_kpts_buffer_  = [device newBufferWithLength:N * sizeof(CornerPoint)
                                             options:MTLResourceStorageModeShared];
    right_kpts_buffer_ = [device newBufferWithLength:N * sizeof(CornerPoint)
                                             options:MTLResourceStorageModeShared];

    // Distance matrix: N_left × N_right × sizeof(ushort)
    dist_matrix_buffer_ = [device newBufferWithLength:N * N * sizeof(uint16_t)
                                              options:MTLResourceStorageModeShared];

    // Output matches
    match_buffer_ = [device newBufferWithLength:N * sizeof(StereoMatchResult)
                                        options:MTLResourceStorageModeShared];

    match_count_buffer_ = [device newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];

    params_buffer_ = [device newBufferWithLength:sizeof(StereoParamsGPU)
                                         options:MTLResourceStorageModeShared];

    ready_ = (hamming_pipeline_ != nil && extract_pipeline_ != nil);
    if (ready_) {
        std::cerr << "[MetalStereo] Ready — max_keypoints=" << N << "\n";
    }
}

std::vector<StereoMatchResult> MetalStereoMatcher::match(
    const std::vector<CornerPoint>& left_kpts,
    const std::vector<ORBDescriptorOutput>& left_desc,
    const std::vector<CornerPoint>& right_kpts,
    const std::vector<ORBDescriptorOutput>& right_desc,
    const MetalStereoCalib& calib)
{
    if (!ready_ || left_kpts.empty() || right_kpts.empty()) return {};

    uint32_t nl = (uint32_t)std::min(left_kpts.size(),  (size_t)max_keypoints_);
    uint32_t nr = (uint32_t)std::min(right_kpts.size(), (size_t)max_keypoints_);

    // Upload keypoints
    memcpy([left_kpts_buffer_ contents],  left_kpts.data(),  nl * sizeof(CornerPoint));
    memcpy([right_kpts_buffer_ contents], right_kpts.data(), nr * sizeof(CornerPoint));

    // Upload descriptors (extract the 8×uint32 desc arrays from ORBDescriptorOutput)
    // ORBDescriptorOutput has desc[8] then angle — we need just desc[8] packed tightly
    {
        uint32_t* dst_l = (uint32_t*)[left_desc_buffer_ contents];
        uint32_t* dst_r = (uint32_t*)[right_desc_buffer_ contents];
        for (uint32_t i = 0; i < nl; i++)
            memcpy(dst_l + i * 8, left_desc[i].desc, 8 * sizeof(uint32_t));
        for (uint32_t i = 0; i < nr; i++)
            memcpy(dst_r + i * 8, right_desc[i].desc, 8 * sizeof(uint32_t));
    }

    // Set params
    StereoParamsGPU params;
    params.n_left        = nl;
    params.n_right       = nr;
    params.max_epipolar  = config_.max_epipolar;
    params.min_disparity = config_.min_disparity;
    params.max_disparity = config_.max_disparity;
    params.max_hamming   = config_.max_hamming;
    params.ratio_thresh  = config_.ratio_thresh;
    params.fx = calib.fx;  params.fy = calib.fy;
    params.cx = calib.cx;  params.cy = calib.cy;
    params.baseline = calib.baseline;
    memcpy([params_buffer_ contents], &params, sizeof(StereoParamsGPU));

    // Reset match counter
    uint32_t zero = 0;
    memcpy([match_count_buffer_ contents], &zero, sizeof(uint32_t));

    // === Dispatch both kernels in one command buffer ===
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    // --- Kernel A: Hamming distance matrix (2D dispatch) ---
    {
        id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
        [enc setComputePipelineState:hamming_pipeline_];
        [enc setBuffer:left_desc_buffer_   offset:0 atIndex:0];
        [enc setBuffer:right_desc_buffer_  offset:0 atIndex:1];
        [enc setBuffer:left_kpts_buffer_   offset:0 atIndex:2];
        [enc setBuffer:right_kpts_buffer_  offset:0 atIndex:3];
        [enc setBuffer:dist_matrix_buffer_ offset:0 atIndex:4];
        [enc setBuffer:params_buffer_      offset:0 atIndex:5];

        MTLSize grid  = MTLSizeMake(nl, nr, 1);
        MTLSize group = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
    }

    // --- Kernel B: Best match extraction (1D dispatch) ---
    {
        id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
        [enc setComputePipelineState:extract_pipeline_];
        [enc setBuffer:dist_matrix_buffer_ offset:0 atIndex:0];
        [enc setBuffer:left_kpts_buffer_   offset:0 atIndex:1];
        [enc setBuffer:right_kpts_buffer_  offset:0 atIndex:2];
        [enc setBuffer:match_buffer_       offset:0 atIndex:3];
        [enc setBuffer:match_count_buffer_ offset:0 atIndex:4];
        [enc setBuffer:params_buffer_      offset:0 atIndex:5];

        MTLSize grid  = MTLSizeMake(nl, 1, 1);
        MTLSize group = MTLSizeMake(std::min(nl, 256u), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    last_gpu_ms_ = ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1000.0;

    // Read results
    last_match_count_ = *(uint32_t*)[match_count_buffer_ contents];
    uint32_t n = std::min(last_match_count_, nl);

    std::vector<StereoMatchResult> result(n);
    if (n > 0) {
        memcpy(result.data(), [match_buffer_ contents], n * sizeof(StereoMatchResult));
    }
    return result;
}

}
