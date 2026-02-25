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

struct ORBDescriptorOutput {
    uint32_t desc[8];   // 256 bits = 8 × uint32
    float    angle;     // orientation in radians
};

struct ORBDescriptorConfig {
    int patch_radius = 15;  // 15 → 31×31 patch
};

class MetalORBDescriptor {
public:
    MetalORBDescriptor(MetalContext* context,
                       const std::string& metallib_path,
                       const ORBDescriptorConfig& config = ORBDescriptorConfig());

    ~MetalORBDescriptor() = default;

    /// Compute descriptors for keypoints. Returns one ORBDescriptorOutput per keypoint.
    std::vector<ORBDescriptorOutput> describe(void* image_texture,
                                              const std::vector<CornerPoint>& keypoints);

    double lastGpuTimeMs() const { return last_gpu_ms_; }
    bool   isReady()       const { return ready_; }

private:
    MetalContext* context_;

    PipelinePtr pipeline_;
    BufferPtr   keypoint_buffer_;
    BufferPtr   output_buffer_;
    BufferPtr   params_buffer_;
    BufferPtr   pattern_buffer_;   // ORB BRIEF test pairs (constant)

    ORBDescriptorConfig config_;
    uint32_t max_keypoints_ = 2000;
    double   last_gpu_ms_ = 0.0;
    bool     ready_ = false;
};

}