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

struct CornerPoint;
struct ORBDescriptorOutput;

struct StereoMatchResult {
    uint32_t left_idx;
    uint32_t right_idx;
    float    disparity;
    float    point_3d[3];   // triangulated (X, Y, Z) in camera frame
};

struct MetalStereoConfig {
    float max_epipolar   = 0.5f;    // pixels
    float min_disparity  = 10.0f;    // pixels
    float max_disparity  = 120.0f;  // pixels
    int   max_hamming    = 50;      // Hamming distance threshold
    float ratio_thresh   = 0.8f;    // Lowe's ratio test
};

struct MetalStereoCalib {
    float fx, fy, cx, cy, baseline;
};

class MetalStereoMatcher {
public:
    MetalStereoMatcher(MetalContext* context,
                       const std::string& metallib_path,
                       const MetalStereoConfig& config = MetalStereoConfig());

    ~MetalStereoMatcher() = default;

    /// Match left vs right keypoints using ORB descriptors
    /// Returns stereo matches with triangulated 3D points
    std::vector<StereoMatchResult> match(
        const std::vector<CornerPoint>& left_kpts,
        const std::vector<ORBDescriptorOutput>& left_desc,
        const std::vector<CornerPoint>& right_kpts,
        const std::vector<ORBDescriptorOutput>& right_desc,
        const MetalStereoCalib& calib);

    uint32_t lastMatchCount() const { return last_match_count_; }
    double   lastGpuTimeMs()  const { return last_gpu_ms_; }
    bool     isReady()        const { return ready_; }

private:
    MetalContext* context_;

    PipelinePtr hamming_pipeline_;
    PipelinePtr extract_pipeline_;

    BufferPtr left_desc_buffer_;
    BufferPtr right_desc_buffer_;
    BufferPtr left_kpts_buffer_;
    BufferPtr right_kpts_buffer_;
    BufferPtr dist_matrix_buffer_;
    BufferPtr match_buffer_;
    BufferPtr match_count_buffer_;
    BufferPtr params_buffer_;

    MetalStereoConfig config_;
    uint32_t max_keypoints_ = 500;
    uint32_t last_match_count_ = 0;
    double   last_gpu_ms_ = 0.0;
    bool     ready_ = false;
};

}
