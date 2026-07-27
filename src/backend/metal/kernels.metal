#include <metal_stdlib>
using namespace metal;

static ulong view_offset(
    ulong logical,
    constant ulong* dims,
    constant ulong* strides,
    uint rank,
    ulong offset
) {
    for (uint axis = rank; axis > 0; --axis) {
        ulong dim = dims[axis - 1];
        if (dim != 0) {
            offset += (logical % dim) * strides[axis - 1];
            logical /= dim;
        }
    }
    return offset;
}

#define COPY_KERNEL(TYPE, NAME) \
kernel void copy_##NAME( \
    device const TYPE* src [[buffer(0)]], \
    device TYPE* dst [[buffer(1)]], \
    constant ulong* dims [[buffer(2)]], \
    constant ulong* strides [[buffer(3)]], \
    constant uint& rank [[buffer(4)]], \
    constant ulong& offset [[buffer(5)]], \
    constant ulong& len [[buffer(6)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid < len) dst[gid] = src[view_offset(gid, dims, strides, rank, offset)]; \
} \
kernel void copy_into_##NAME( \
    device const TYPE* src [[buffer(0)]], \
    device TYPE* dst [[buffer(1)]], \
    constant ulong* src_dims [[buffer(2)]], \
    constant ulong* src_strides [[buffer(3)]], \
    constant uint& src_rank [[buffer(4)]], \
    constant ulong& src_offset [[buffer(5)]], \
    constant ulong* dst_dims [[buffer(6)]], \
    constant ulong* dst_strides [[buffer(7)]], \
    constant uint& dst_rank [[buffer(8)]], \
    constant ulong& dst_offset [[buffer(9)]], \
    constant ulong& len [[buffer(10)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid < len) { \
        ulong from = view_offset(gid, src_dims, src_strides, src_rank, src_offset); \
        ulong to = view_offset(gid, dst_dims, dst_strides, dst_rank, dst_offset); \
        dst[to] = src[from]; \
    } \
}

COPY_KERNEL(half, f16)
COPY_KERNEL(float, f32)
COPY_KERNEL(long, i64)
COPY_KERNEL(uchar, bool)
