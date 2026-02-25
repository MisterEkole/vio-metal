#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace vio { class MetalContext; }

#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLComputePipelineState> PipelinePtr;
typedef id<MTLBuffer>              BufferPtr;
#else
typedef void* PipelinePtr;
typedef void* BufferPtr;
#endif

namespace vio {

struct CornerPoint;  // forward — defined in MetalFASTDetector.h

struct HarrisConfig {
    int   patch_radius = 3;     // 3 → 7×7 window
    float k            = 0.04f; // Harris parameter
};

class MetalHarrisResponse {
public:
    MetalHarrisResponse(MetalContext* context,
                        const std::string& metallib_path,
                        const HarrisConfig& config = HarrisConfig());

    ~MetalHarrisResponse() = default;

    /// Score corners in-place (updates response field)
    /// image_texture: the grayscale R8Unorm texture
    /// corners: output from FAST detector (modified in-place)
    void score(void* image_texture,
               std::vector<CornerPoint>& corners);

    double lastGpuTimeMs() const { return last_gpu_ms_; }
    bool   isReady()       const { return ready_; }

private:
    MetalContext* context_;

    PipelinePtr pipeline_;
    BufferPtr   corner_buffer_;
    BufferPtr   params_buffer_;

    HarrisConfig config_;
    uint32_t max_corners_ = 50000;
    double   last_gpu_ms_ = 0.0;
    bool     ready_ = false;
};

}
