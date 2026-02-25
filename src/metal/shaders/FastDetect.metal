#include <metal_stdlib>
using namespace metal;

// Output corner struct — matches the CPU-side definition
struct CornerPoint {
    float2 position;       // pixel coordinates
    float  response;       // filled later by Harris (0 for now)
    uint   pyramid_level;  // which scale level
};

struct FASTParams {
    int  threshold;     // intensity diff threshold (typically 20)
    uint max_corners;   // output buffer capacity
    uint width;
    uint height;
};

// Bresenham circle of radius 3: 16 pixels
// Ordered clockwise from 12 o'clock
constant int2 circle[16] = {
    {0,-3}, {1,-3}, {2,-2}, {3,-1},   //  0- 3 (top-right arc)
    {3,0},  {3,1},  {2,2},  {1,3},    //  4- 7 (right-bottom arc)
    {0,3},  {-1,3}, {-2,2}, {-3,1},   //  8-11 (bottom-left arc)
    {-3,0}, {-3,-1},{-2,-2},{-1,-3}   // 12-15 (left-top arc)
};

kernel void fast_detect(
    texture2d<float, access::read>  image      [[texture(0)]],
    device CornerPoint*             corners    [[buffer(0)]],
    device atomic_uint*             count      [[buffer(1)]],
    constant FASTParams&            params     [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // --- Bounds: skip 3px border (circle radius) ---
    if (gid.x < 3 || gid.x >= params.width - 3 ||
        gid.y < 3 || gid.y >= params.height - 3) return;

    // Read center pixel intensity (R8Unorm is [0,1], scale to [0,255])
    float center = image.read(gid).r * 255.0;
    float t    = float(params.threshold);
    float high = center + t;
    float low  = center - t;

    // === High-speed test: compass points 0, 4, 8, 12 ===
    // If at least 3 of these 4 aren't consistently bright or dark,
    // there can't be 9 contiguous → reject immediately.
    // This eliminates ~80% of pixels in 4 reads instead of 16.
    float p0  = image.read(uint2(gid.x + circle[0].x,  gid.y + circle[0].y)).r  * 255.0;
    float p4  = image.read(uint2(gid.x + circle[4].x,  gid.y + circle[4].y)).r  * 255.0;
    float p8  = image.read(uint2(gid.x + circle[8].x,  gid.y + circle[8].y)).r  * 255.0;
    float p12 = image.read(uint2(gid.x + circle[12].x, gid.y + circle[12].y)).r * 255.0;

    int n_bright = int(p0 > high) + int(p4 > high) + int(p8 > high) + int(p12 > high);
    int n_dark   = int(p0 < low)  + int(p4 < low)  + int(p8 < low)  + int(p12 < low);

    if (n_bright < 2 && n_dark < 2) return;

    // === Full test: read all 16 circle pixels ===
    float vals[16];
    for (int i = 0; i < 16; i++) {
        vals[i] = image.read(uint2(gid.x + circle[i].x, gid.y + circle[i].y)).r * 255.0;
    }

    // Check for 9 contiguous pixels all brighter or all darker
    // We scan the 16-pixel ring once for bright and once for dark,
    // tracking the longest consecutive run. Because the ring wraps,
    // we scan 16+8 positions (enough to catch any 9-run that straddles index 15→0).
    bool is_corner = false;

    if (n_bright >= 2) {
        int run = 0;
        for (int k = 0; k < 25 && !is_corner; k++) {   // 16 + 9 - 1 = 24, use 25 for safety
            if (vals[k % 16] > high) {
                run++;
                if (run >= 9) is_corner = true;
            } else {
                run = 0;
            }
        }
    }

    if (!is_corner && n_dark >= 2) {
        int run = 0;
        for (int k = 0; k < 25 && !is_corner; k++) {
            if (vals[k % 16] < low) {
                run++;
                if (run >= 9) is_corner = true;
            } else {
                run = 0;
            }
        }
    }

    if (!is_corner) return;

    // === Corner score: sum of absolute differences from threshold ===
    // (Quick score — Harris replaces this later, but useful for early NMS)
    float score = 0;
    for (int i = 0; i < 16; i++) {
        float diff = abs(vals[i] - center) - t;
        if (diff > 0) score += diff;
    }

    // === Atomic append to output buffer ===
    uint slot = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (slot < params.max_corners) {
        corners[slot].position = float2(gid);
        corners[slot].response = score;
        corners[slot].pyramid_level = 0;   // caller sets this for multi-scale
    }
}
