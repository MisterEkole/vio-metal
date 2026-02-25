#import "MetalUndistort.h"
#import "MetalContext.h"
#import <Metal/Metal.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace vio {

MetalUndistort::MetalUndistort(MetalContext* context,
                               const cv::Mat& map_x,
                               const cv::Mat& map_y,
                               int width, int height,
                               const std::string& metallib_path)
    : context_(context), width_(width), height_(height)
{
    void* lib_ptr = context_->loadLibrary(metallib_path);
    if (!lib_ptr) {
        std::cerr << "[MetalUndistort] Failed to load metallib: " << metallib_path << "\n";
        return;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib_ptr;
    
    void* pipe_ptr = context_->getPipeline("undistort", (__bridge void*)library);
    pipeline_ = (__bridge id<MTLComputePipelineState>)pipe_ptr;
    
    id<MTLDevice> device = (__bridge id<MTLDevice>)context_->getDevice();

    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    
    map_x_texture_ = [device newTextureWithDescriptor:desc];
    map_y_texture_ = [device newTextureWithDescriptor:desc];
    
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [map_x_texture_ replaceRegion:region mipmapLevel:0 withBytes:map_x.data bytesPerRow:width * sizeof(float)];
    [map_y_texture_ replaceRegion:region mipmapLevel:0 withBytes:map_y.data bytesPerRow:width * sizeof(float)];

    desc.pixelFormat = MTLPixelFormatR8Unorm;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    input_texture_ = [device newTextureWithDescriptor:desc];
    output_texture_ = [device newTextureWithDescriptor:desc];

    ready_ = (pipeline_ != nil && map_x_texture_ != nil);
}

// ASYNCHRONOUS DISPATCH: Call at start of frame
void MetalUndistort::encodeUndistort(const cv::Mat& input) {
    if (!ready_ || input.empty()) return;

    // Upload CPU image to GPU texture
    MTLRegion region = MTLRegionMake2D(0, 0, width_, height_);
    [input_texture_ replaceRegion:region mipmapLevel:0 withBytes:input.data bytesPerRow:input.step];

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context_->getCommandQueue();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline_];
    [encoder setTexture:input_texture_  atIndex:0];
    [encoder setTexture:map_x_texture_  atIndex:1];
    [encoder setTexture:map_y_texture_  atIndex:2];
    [encoder setTexture:output_texture_ atIndex:3];

    MTLSize gridSize = MTLSizeMake(width_, height_, 1);
    MTLSize groupSize = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];
    
    // Fire the GPU. No wait!!

    [commandBuffer commit];
    context_->setLastBuffer((__bridge void*)commandBuffer);
}

// DATA RETRIEVAL: call after end frame 
cv::Mat MetalUndistort::getOutputMat() {
    cv::Mat output(height_, width_, CV_8UC1);
    MTLRegion region = MTLRegionMake2D(0, 0, width_, height_);
    
    
    [output_texture_ getBytes:output.data bytesPerRow:output.step fromRegion:region mipmapLevel:0];
    return output;
}


cv::Mat MetalUndistort::undistort(const cv::Mat& input) {
    encodeUndistort(input);
    context_->waitForGPU(); // Force wait 
    return getOutputMat();
}

}