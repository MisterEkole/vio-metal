#include <metal_stdlib>
using namespace metal;

struct CornerPoint {
    float2   position;
    float    response;
    uint     pyramid_level;
};

struct StereoMatchResult {
    uint   left_idx;
    uint   right_idx;
    float  disparity;
    float3 point_3d;     // triangulated in left camera frame
};

struct StereoParams {
    uint   n_left;
    uint   n_right;
    float  max_epipolar;    // max y-difference (2.0 px for rectified)
    float  min_disparity;   // 1.0
    float  max_disparity;   // 120.0
    uint   max_hamming;     // 50
    float  ratio_thresh;    // 0.8
    float  fx, fy, cx, cy, baseline;
};

// ================================================================
// Kernel A: Compute Hamming distance matrix (2D dispatch)
// One thread per (left_idx, right_idx) pair
// ================================================================
kernel void stereo_hamming(
    device uint32_t*        left_desc    [[buffer(0)]],  // n_left × 8 uint32
    device uint32_t*        right_desc   [[buffer(1)]],  // n_right × 8 uint32
    device CornerPoint*     left_kpts    [[buffer(2)]],
    device CornerPoint*     right_kpts   [[buffer(3)]],
    device ushort*          dist_matrix  [[buffer(4)]],  // n_left × n_right
    constant StereoParams&  params       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint li = gid.x;
    uint ri = gid.y;
    if (li >= params.n_left || ri >= params.n_right) return;

    ushort result = 0xFFFF;  // sentinel = invalid

    // Epipolar constraint: y-coords must match on rectified images
    float dy = abs(left_kpts[li].position.y - right_kpts[ri].position.y);
    if (dy <= params.max_epipolar) {
        // Disparity constraint: left.x > right.x (object closer = larger disparity)
        float d = left_kpts[li].position.x - right_kpts[ri].position.x;
        if (d >= params.min_disparity && d <= params.max_disparity) {
            // Hamming distance: XOR each 32-bit word, popcount the result
            uint dist = 0;
            for (int w = 0; w < 8; w++) {
                dist += popcount(left_desc[li * 8 + w] ^ right_desc[ri * 8 + w]);
            }
            result = ushort(dist);
        }
    }

    dist_matrix[li * params.n_right + ri] = result;
}

// ================================================================
// Kernel B: Extract best match per left keypoint (1D dispatch)
// ================================================================
kernel void stereo_extract(
    device ushort*              dist_matrix  [[buffer(0)]],
    device CornerPoint*         left_kpts    [[buffer(1)]],
    device CornerPoint*         right_kpts   [[buffer(2)]],
    device StereoMatchResult*   matches      [[buffer(3)]],
    device atomic_uint*         match_count  [[buffer(4)]],
    constant StereoParams&      params       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_left) return;

    ushort best = 0xFFFF;
    ushort second = 0xFFFF;
    uint   best_ri = 0;

    // Scan all right keypoints for best and second-best match
    for (uint ri = 0; ri < params.n_right; ri++) {
        ushort d = dist_matrix[tid * params.n_right + ri];
        if (d < best) {
            second = best;
            best = d;
            best_ri = ri;
        } else if (d < second) {
            second = d;
        }
    }

    // Distance threshold
    if (best > ushort(params.max_hamming)) return;

    // Lowe's ratio test: best must be significantly better than second-best
    // if (second < 0xFFFF && second > 0) {
    //     if (float(best) / float(second) > params.ratio_thresh) return;
    // }

    // Triangulate: depth = fx * baseline / disparity
    float disp = left_kpts[tid].position.x - right_kpts[best_ri].position.x;
    float Z = params.fx * params.baseline / disp;
    float X = Z * (left_kpts[tid].position.x - params.cx) / params.fx;
    float Y = Z * (left_kpts[tid].position.y - params.cy) / params.fy;

    // Atomic append to output
    uint slot = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
    matches[slot].left_idx  = tid;
    matches[slot].right_idx = best_ri;
    matches[slot].disparity = disp;
    matches[slot].point_3d  = float3(X, Y, Z);
}
