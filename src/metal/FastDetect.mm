#import "FastDetect.h"
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <iostream>

namespace vio {

struct FastParamsGPU {
    int      threshold;
    uint32_t max_corners;
    uint32_t width;
    uint32_t height;
};

MetalFastDetector::MetalFastDetector(MetalContext* context,
                                     int width, int height,
                                     const std::string& metallib_path,
                                     const FastDetectorConfig& config)
    : context_(context), width_(width), height_(height), config_(config)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalFastDetector] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;

    void* pipe_ptr = context_->getPipeline("fast_detect", (__bridge void*)library);
    pipeline_ = (__bridge id<MTLComputePipelineState>)pipe_ptr;

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    corner_buffer_ = [device newBufferWithLength:config_.max_corners * sizeof(CornerPoint)
                                         options:MTLResourceStorageModeShared];
    count_buffer_ = [device newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];

    FastParamsGPU params;
    params.threshold  = config_.threshold;
    params.max_corners = config_.max_corners;
    params.width      = width_;
    params.height     = height_;

    params_buffer_ = [device newBufferWithLength:sizeof(FastParamsGPU)
                                         options:MTLResourceStorageModeShared];
    memcpy([params_buffer_ contents], &params, sizeof(FastParamsGPU));

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                          width:width_
                                                         height:height_
                                                        mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    input_texture_ = [device newTextureWithDescriptor:desc];

    ready_ = (pipeline_ != nil && input_texture_ != nil);
    if (ready_) {
        std::cerr << "[MetalFastDetector] Ready — threshold=" << config_.threshold
                  << " max_corners=" << config_.max_corners << "\n";
    }
}

std::vector<CornerPoint> MetalFastDetector::detect(const uint8_t* image_data, int stride) {
    if (!ready_) return {};

    MTLRegion region = MTLRegionMake2D(0, 0, width_, height_);
    [input_texture_ replaceRegion:region
                      mipmapLevel:0
                        withBytes:image_data
                      bytesPerRow:stride];
    return dispatchAndRead((__bridge void*)input_texture_);
}

std::vector<CornerPoint> MetalFastDetector::detect(void* texture_ptr) {
    if (!ready_ || !texture_ptr) return {};
    return dispatchAndRead(texture_ptr);
}

std::vector<CornerPoint> MetalFastDetector::dispatchAndRead(void* texture_ptr) {
    id<MTLTexture> texture = (__bridge id<MTLTexture>)texture_ptr;

    uint32_t zero = 0;
    memcpy([count_buffer_ contents], &zero, sizeof(uint32_t));

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:texture       atIndex:0];
    [encoder setBuffer:corner_buffer_ offset:0 atIndex:0];
    [encoder setBuffer:count_buffer_  offset:0 atIndex:1];
    [encoder setBuffer:params_buffer_ offset:0 atIndex:2];

    MTLSize gridSize  = MTLSizeMake(width_, height_, 1);
    MTLSize groupSize = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    last_gpu_ms_ = ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1000.0;

    last_count_ = *(uint32_t*)[count_buffer_ contents];
    uint32_t n = std::min(last_count_, (uint32_t)config_.max_corners);

    std::vector<CornerPoint> result(n);
    if (n > 0) {
        memcpy(result.data(), [corner_buffer_ contents], n * sizeof(CornerPoint));
    }
    return result;
}

}