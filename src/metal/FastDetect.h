#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace vio{class MetalContext;}

#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLComputePipelineState> PipelinePtr;
typedef id<MTLBuffer>               BufferPtr;
typedef id<MTLTexture>              TexturePtr;

#else
typedef void* PipelinePtr;
typedef void* BufferPtr;
typedef void* TexturePtr;
#endif    

namespace vio{

    struct CornerPoint{
        float position[2];
        float response;
        uint32_t pyramid_level;
    };

    struct FastDetectorConfig{
        int threshold = 20;
        int max_corners = 50000;
    };

    class MetalFastDetector{
        public:
        MetalFastDetector(
            MetalContext* context,
            int width, int height,
            const std::string& metallib_path,
            const FastDetectorConfig& config = FastDetectorConfig());
        ~MetalFastDetector() = default;

        std::vector<CornerPoint> detect( const uint8_t* image_data, int stride ); // run fastdetect on grayscale R8Unorm texture and return detected corner(pos+score+level)
        std::vector<CornerPoint> detect(void* input_texture);
        uint32_t lastCornerCount() const {return last_count_;}
        double lastGpuTimeMs() const {return last_gpu_ms_;}
        bool    isReady() const {return ready_;}


        private:

        MetalContext* context_;
        PipelinePtr pipeline_;
        BufferPtr corner_buffer_;  // output: detected corners
        BufferPtr count_buffer_;    // output: atomic counts
        BufferPtr params_buffer_;   // input: FastParams uniform
        TexturePtr input_texture_;  // image

        int width_,height_;
        FastDetectorConfig config_;
        uint32_t last_count_ = 0;
        double last_gpu_ms_ = 0.0;
        bool ready_ = false;

        std::vector<CornerPoint> dispatchAndRead(void* texture);
    


    };
}