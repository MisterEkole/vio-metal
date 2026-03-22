#import "ORBDescriptor.h"
#import "FastDetect.h"   // for CornerPoint
#import "MetalContext.h"
#import <Metal/Metal.h>
#include "ORBPattern.h"         // the 256×4 BRIEF test pairs
#include <iostream>

namespace vio {

struct ORBParamsGPU {
    uint32_t n_keypoints;
    uint32_t patch_radius;
};

MetalORBDescriptor::MetalORBDescriptor(MetalContext* context,
                                       const std::string& metallib_path,
                                       const ORBDescriptorConfig& config)
    : context_(context), config_(config)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalORB] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;

    void* pipe_ptr = context_->getPipeline("orb_describe", (__bridge void*)library);
    pipeline_ = (__bridge id<MTLComputePipelineState>)pipe_ptr;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    keypoint_buffer_ = [device newBufferWithLength:max_keypoints_ * sizeof(CornerPoint)
                                           options:MTLResourceStorageModeShared];

    output_buffer_ = [device newBufferWithLength:max_keypoints_ * sizeof(ORBDescriptorOutput)
                                         options:MTLResourceStorageModeShared];

    params_buffer_ = [device newBufferWithLength:sizeof(ORBParamsGPU)
                                         options:MTLResourceStorageModeShared];

    pattern_buffer_ = [device newBufferWithLength:sizeof(ORB_PATTERN)
                                          options:MTLResourceStorageModeShared];
    memcpy([pattern_buffer_ contents], ORB_PATTERN, sizeof(ORB_PATTERN));

    ready_ = (pipeline_ != nil && pattern_buffer_ != nil);
    if (ready_) {
        std::cerr << "[MetalORB] Ready — 256-bit rotated BRIEF\n";
    }
}

std::vector<ORBDescriptorOutput> MetalORBDescriptor::describe(
    void* image_texture,
    const std::vector<CornerPoint>& keypoints)
{
    if (!ready_ || keypoints.empty()) return {};

    uint32_t n = (uint32_t)std::min(keypoints.size(), (size_t)max_keypoints_);
    id<MTLTexture> texture = (__bridge id<MTLTexture>)image_texture;

    memcpy([keypoint_buffer_ contents], keypoints.data(), n * sizeof(CornerPoint));

    ORBParamsGPU params;
    params.n_keypoints  = n;
    params.patch_radius = config_.patch_radius;
    memcpy([params_buffer_ contents], &params, sizeof(ORBParamsGPU));

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:texture          atIndex:0];
    [encoder setBuffer:keypoint_buffer_  offset:0 atIndex:0];
    [encoder setBuffer:output_buffer_    offset:0 atIndex:1];
    [encoder setBuffer:params_buffer_    offset:0 atIndex:2];
    [encoder setBuffer:pattern_buffer_   offset:0 atIndex:3];

    MTLSize gridSize  = MTLSizeMake(n, 1, 1);
    MTLSize groupSize = MTLSizeMake(std::min(n, 256u), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    last_gpu_ms_ = ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1000.0;

    std::vector<ORBDescriptorOutput> result(n);
    memcpy(result.data(), [output_buffer_ contents], n * sizeof(ORBDescriptorOutput));
    return result;
}

}
