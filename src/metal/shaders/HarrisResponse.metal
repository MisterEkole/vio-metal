#include <metal_stdlib>
using namespace metal;

struct CornerPoint {
    float2   position;
    float    response;
    uint     pyramid_level;
};

struct HarrisParams {
    uint  n_corners;     // number of input corners to score
    int   patch_radius;  // half-size of integration window (3 → 7×7)
    float k;             // Harris parameter (typically 0.04)
};

kernel void harris_response(
    texture2d<float, access::read>  image    [[texture(0)]],
    device CornerPoint*             corners  [[buffer(0)]],
    constant HarrisParams&          params   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_corners) return;

    int cx = int(corners[tid].position.x);
    int cy = int(corners[tid].position.y);
    int R  = params.patch_radius;

    // Build structure tensor over a (2R+1)×(2R+1) patch
    // using central-difference gradients: Ix = I(x+1) - I(x-1), etc.
    float sxx = 0.0, sxy = 0.0, syy = 0.0;

    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            int px = cx + dx;
            int py = cy + dy;

            // Central difference gradients (scaled by 255 for numerical range)
            float ix = image.read(uint2(px + 1, py)).r - image.read(uint2(px - 1, py)).r;
            float iy = image.read(uint2(px, py + 1)).r - image.read(uint2(px, py - 1)).r;

            sxx += ix * ix;
            sxy += ix * iy;
            syy += iy * iy;
        }
    }

    // Harris corner response: R = det(M) - k * trace(M)^2
    float det   = sxx * syy - sxy * sxy;
    float trace = sxx + syy;
    corners[tid].response = det - params.k * trace * trace;
}
