#include <metal_stdlib>
using namespace metal;

struct KLTParams {
    uint   n_points;
    uint   max_iterations;
    float  epsilon;
    int    win_radius;
    uint   max_level;
    float  min_eigenvalue;
};

// Custom 32-bit Float Bilinear Interpolation
inline float sample_bilinear_manual(texture2d<float, access::read> tex, float2 uv) {
    int x0 = int(floor(uv.x));
    int y0 = int(floor(uv.y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp to boundaries safely
    int w = tex.get_width() - 1;
    int h = tex.get_height() - 1;
    x0 = clamp(x0, 0, w); x1 = clamp(x1, 0, w);
    y0 = clamp(y0, 0, h); y1 = clamp(y1, 0, h);
    
    // Read raw 8-bit values but store as 32-bit float
    float p00 = tex.read(uint2(x0, y0)).r;
    float p10 = tex.read(uint2(x1, y0)).r;
    float p01 = tex.read(uint2(x0, y1)).r;
    float p11 = tex.read(uint2(x1, y1)).r;
    
    float tx = uv.x - floor(uv.x);
    float ty = uv.y - floor(uv.y);
    
    float top = mix(p00, p10, tx);
    float bottom = mix(p01, p11, tx);
    return mix(top, bottom, ty);
}

// Helper to route the manual sampler
inline float manual_sample_level(
    texture2d<float, access::read> t0,
    texture2d<float, access::read> t1,
    texture2d<float, access::read> t2,
    texture2d<float, access::read> t3,
    int level, float2 pos)
{
    switch (level) {
        case 0:  return sample_bilinear_manual(t0, pos);
        case 1:  return sample_bilinear_manual(t1, pos);
        case 2:  return sample_bilinear_manual(t2, pos);
        default: return sample_bilinear_manual(t3, pos);
    }
}

kernel void klt_track_forward(
    // NOTE: Changed from access::sample to access::read
    texture2d<float, access::read> prev0 [[texture(0)]],
    texture2d<float, access::read> prev1 [[texture(1)]],
    texture2d<float, access::read> prev2 [[texture(2)]],
    texture2d<float, access::read> prev3 [[texture(3)]],
    texture2d<float, access::read> curr0 [[texture(4)]],
    texture2d<float, access::read> curr1 [[texture(5)]],
    texture2d<float, access::read> curr2 [[texture(6)]],
    texture2d<float, access::read> curr3 [[texture(7)]],
    device float2* prev_pts  [[buffer(0)]],
    device float2* curr_pts  [[buffer(1)]],
    device uint8_t* status    [[buffer(2)]],
    constant KLTParams&   params    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_points) return;

    float2 prev_pt = prev_pts[tid];
    float2 flow = float2(0.0);
    int R = params.win_radius;
    bool lost = false;

    for (int level = int(params.max_level); level >= 0; level--) {
        float scale = 1.0 / float(1 << level);
        float2 prev_scaled = prev_pt * scale;
        float2 guess = prev_scaled + flow;

        float gxx = 0.0, gxy = 0.0, gyy = 0.0;

        for (int dy = -R; dy <= R; dy++) {
            for (int dx = -R; dx <= R; dx++) {
                float2 p = prev_scaled + float2(dx, dy);

                // Replaced with manual 32-bit interpolation
                float ix = manual_sample_level(prev0, prev1, prev2, prev3, level, p + float2(1, 0))
                         - manual_sample_level(prev0, prev1, prev2, prev3, level, p - float2(1, 0));
                float iy = manual_sample_level(prev0, prev1, prev2, prev3, level, p + float2(0, 1))
                         - manual_sample_level(prev0, prev1, prev2, prev3, level, p - float2(0, 1));

                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
            }
        }

        float det = gxx * gyy - gxy * gxy;
        float trace = gxx + gyy;
        float disc = max(trace * trace * 0.25 - det, 0.0);
        float min_eig = trace * 0.5 - sqrt(disc);

        if (min_eig < params.min_eigenvalue || abs(det) < 1e-10) {
            lost = true;
            break;
        }

        float inv_det = 1.0 / det;
        float2 local_flow = float2(0.0);

        for (uint iter = 0; iter < params.max_iterations; iter++) {
            float bx = 0.0, by = 0.0;

            for (int dy = -R; dy <= R; dy++) {
                for (int dx = -R; dx <= R; dx++) {
                    float2 p = prev_scaled + float2(dx, dy);
                    float2 q = guess + local_flow + float2(dx, dy);

                    float ix = manual_sample_level(prev0, prev1, prev2, prev3, level, p + float2(1, 0))
                             - manual_sample_level(prev0, prev1, prev2, prev3, level, p - float2(1, 0));
                    float iy = manual_sample_level(prev0, prev1, prev2, prev3, level, p + float2(0, 1))
                             - manual_sample_level(prev0, prev1, prev2, prev3, level, p - float2(0, 1));

                    
                    float it = manual_sample_level(prev0, prev1, prev2, prev3, level, p)
                             - manual_sample_level(curr0, curr1, curr2, curr3, level, q);

                    bx += ix * it;
                    by += iy * it;
                }
            }

            float2 delta;
            delta.x = inv_det * ( gyy * bx - gxy * by);
            delta.y = inv_det * (-gxy * bx + gxx * by);
            local_flow += delta;

            if (length(delta) < params.epsilon) break;
        }

        flow = (local_flow + guess - prev_scaled);
        if (level > 0) flow *= 2.0;
    }

    float2 result = prev_pt + flow;
    float w = float(curr0.get_width());
    float h = float(curr0.get_height());
    
    if (lost || result.x < 0 || result.x >= w || result.y < 0 || result.y >= h) {
        status[tid] = 0;
        curr_pts[tid] = prev_pt;
    } else {
        status[tid] = 1;
        curr_pts[tid] = result;
    }
}