#import "HarrisResponse.h"
#import "FastDetect.h"   // for CornerPoint definition
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <iostream>

namespace vio {

struct HarrisParamsGPU {
    uint32_t n_corners;
    int32_t  patch_radius;
    float    k;
};

MetalHarrisResponse::MetalHarrisResponse(MetalContext* context,
                                         const std::string& metallib_path,
                                         const HarrisConfig& config)
    : context_(context), config_(config)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalHarris] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;

    void* pipe_ptr = context_->getPipeline("harris_response", (__bridge void*)library);
    pipeline_ = (__bridge id<MTLComputePipelineState>)pipe_ptr;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    corner_buffer_ = [device newBufferWithLength:max_corners_ * sizeof(CornerPoint)
                                         options:MTLResourceStorageModeShared];

    params_buffer_ = [device newBufferWithLength:sizeof(HarrisParamsGPU)
                                         options:MTLResourceStorageModeShared];

    ready_ = (pipeline_ != nil && corner_buffer_ != nil);
    if (ready_) {
        std::cerr << "[MetalHarris] Ready — patch=" << (2*config_.patch_radius+1)
                  << "x" << (2*config_.patch_radius+1)
                  << " k=" << config_.k << "\n";
    }
}

void MetalHarrisResponse::score(void* image_texture,
                                std::vector<CornerPoint>& corners)
{
    if (!ready_ || corners.empty()) return;

    uint32_t n = (uint32_t)std::min(corners.size(), (size_t)max_corners_);
    id<MTLTexture> texture = (__bridge id<MTLTexture>)image_texture;

    memcpy([corner_buffer_ contents], corners.data(), n * sizeof(CornerPoint));

    HarrisParamsGPU params;
    params.n_corners    = n;
    params.patch_radius = config_.patch_radius;
    params.k            = config_.k;
    memcpy([params_buffer_ contents], &params, sizeof(HarrisParamsGPU));

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:texture        atIndex:0];
    [encoder setBuffer:corner_buffer_  offset:0 atIndex:0];
    [encoder setBuffer:params_buffer_  offset:0 atIndex:1];

    MTLSize gridSize  = MTLSizeMake(n, 1, 1);
    MTLSize groupSize = MTLSizeMake(std::min(n, 256u), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    last_gpu_ms_ = ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1000.0;

    memcpy(corners.data(), [corner_buffer_ contents], n * sizeof(CornerPoint));
}

}
