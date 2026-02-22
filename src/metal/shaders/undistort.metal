#include <metal_stdlib>
using namespace metal;

kernel void undistort(
    texture2d<float, access::sample> input     [[texture(0)]],
    texture2d<float, access::read>   map_x     [[texture(1)]],
    texture2d<float, access::read>   map_y     [[texture(2)]],
    texture2d<float, access::write>  output    [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    float2 src_coord = float2(
        map_x.read(gid).r,
        map_y.read(gid).r
    );

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    float4 pixel = input.sample(s, src_coord);
    output.write(pixel, gid);
}