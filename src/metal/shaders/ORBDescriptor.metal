#include <metal_stdlib>
using namespace metal;

struct CornerPoint {
    float2   position;
    float    response;
    uint     pyramid_level;
};

struct ORBOutput {
    uint32_t desc[8];    // 256 bits = 8 × uint32
    float    angle;      // orientation in radians
};

struct ORBParams {
    uint n_keypoints;
    uint patch_radius;   // 15 for standard ORB (31×31 patch)
};

// Row extents for circular patch: u_max[|dy|] = floor(sqrt(R² - dy²))
// Precomputed for radius 15
constant int u_max[16] = {15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3};

kernel void orb_describe(
    texture2d<float, access::read>   image       [[texture(0)]],
    device CornerPoint*              keypoints   [[buffer(0)]],
    device ORBOutput*                output      [[buffer(1)]],
    constant ORBParams&              params      [[buffer(2)]],
    constant int*                    pattern     [[buffer(3)]],  // 256×4 ints from ORBPattern.h
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_keypoints) return;

    int cx = int(keypoints[tid].position.x);
    int cy = int(keypoints[tid].position.y);

    // ================================================================
    // Phase 1: Intensity centroid orientation
    // Compute moments m10, m01 over a circular patch of radius 15
    // θ = atan2(m01, m10)
    // ================================================================
    float m10 = 0.0, m01 = 0.0;

    for (int dy = -15; dy <= 15; dy++) {
        int max_dx = u_max[abs(dy)];
        for (int dx = -max_dx; dx <= max_dx; dx++) {
            float val = image.read(uint2(cx + dx, cy + dy)).r;
            m10 += float(dx) * val;
            m01 += float(dy) * val;
        }
    }

    float angle = atan2(m01, m10);
    output[tid].angle = angle;

    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // ================================================================
    // Phase 2: Rotated BRIEF descriptor
    // For each of 256 test pairs, rotate by θ, compare intensities,
    // pack result into 8 × uint32 (256 bits)
    // ================================================================
    for (int word = 0; word < 8; word++) {
        uint32_t bits = 0;

        for (int bit = 0; bit < 32; bit++) {
            int idx = (word * 32 + bit) * 4;

            // Read the unrotated test pair offsets
            float dx1 = float(pattern[idx]);
            float dy1 = float(pattern[idx + 1]);
            float dx2 = float(pattern[idx + 2]);
            float dy2 = float(pattern[idx + 3]);

            // Rotate by keypoint orientation
            int rx1 = int(cos_a * dx1 - sin_a * dy1);
            int ry1 = int(sin_a * dx1 + cos_a * dy1);
            int rx2 = int(cos_a * dx2 - sin_a * dy2);
            int ry2 = int(sin_a * dx2 + cos_a * dy2);

            float I1 = image.read(uint2(cx + rx1, cy + ry1)).r;
            float I2 = image.read(uint2(cx + rx2, cy + ry2)).r;

            if (I1 < I2) bits |= (1u << bit);
        }

        output[tid].desc[word] = bits;
    }
}
