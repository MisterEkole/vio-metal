#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// Forward declare the MetalContext so we don't need to import its Obj-C headers here
namespace vio { class MetalContext; }

// Use this macro to hide Objective-C types from pure C++ files
#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLComputePipelineState> PipelinePtr;
typedef id<MTLTexture> TexturePtr;
#else
typedef void* PipelinePtr;
typedef void* TexturePtr;
#endif

namespace vio {

class MetalUndistort {
public:
    MetalUndistort(MetalContext* context,
                   const cv::Mat& map_x,
                   const cv::Mat& map_y,
                   int width, int height,
                   const std::string& metallib_path);

    ~MetalUndistort() = default;

    cv::Mat undistort(const cv::Mat& input);

    double lastGpuTimeMs() const { return last_gpu_ms_; }
    bool isReady() const { return ready_; }

private:
    MetalContext* context_; 
    
    // These are now safe for both C++ and Obj-C++ files
    PipelinePtr pipeline_;
    TexturePtr map_x_texture_;
    TexturePtr map_y_texture_;
    TexturePtr input_texture_;
    TexturePtr output_texture_;
    
    int width_, height_;
    double last_gpu_ms_ = 0.0;
    bool ready_ = false;
};

} 