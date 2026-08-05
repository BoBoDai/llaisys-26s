#include "rope_nvidia.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ __forceinline__ float _to_f32(const T &v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else if constexpr (std::is_same_v<T, __half>) return __half2float(v);
    else return __bfloat162float(v);
}
template <typename T>
__device__ __forceinline__ T _from_f32(float v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else if constexpr (std::is_same_v<T, __half>) return __float2half(v);
    else return __float2bfloat16(v);
}

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                             size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;
    size_t seq_stride = nhead * d;
    size_t total = seqlen * nhead * half_d;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t j = flat % half_d;
        size_t rem = flat / half_d;
        size_t h = rem % nhead;
        size_t i = rem / nhead;

        float pos = (float)pos_ids[i];
        float freq = pos / powf(theta, 2.0f * (float)j / (float)d);
        float c = cosf(freq);
        float s = sinf(freq);

        size_t off = i * seq_stride + h * d;
        float a = _to_f32(in[off + j]);
        float b = _to_f32(in[off + j + half_d]);

        out[off + j] = _from_f32<T>(a * c - b * s);
        out[off + j + half_d] = _from_f32<T>(b * c + a * s);
    }
}

template <typename T>
void rope_launch(T *out, const T *in, const int64_t *pos_ids, float theta,
                  size_t seqlen, size_t nhead, size_t d) {
    size_t total = seqlen * nhead * (d / 2);
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    if (grid < 1) grid = 1;
    rope_kernel<T><<<grid, block>>>(out, in, pos_ids, theta, seqlen, nhead, d);
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seqlen = out->shape()[0];
    size_t nhead = out->shape()[1];
    size_t d = out->shape()[2];
    auto dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
      return rope_launch(
        reinterpret_cast<float *>(out->data()), 
        reinterpret_cast<const float *>(in->data()), 
        reinterpret_cast<const int64_t *>(pos_ids->data()), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16: 
     return rope_launch(
        reinterpret_cast<__half *>(out->data()), 
     reinterpret_cast<const __half *>(in->data()), 
     reinterpret_cast<const int64_t *>(pos_ids->data()), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
     return rope_launch(
        reinterpret_cast<__nv_bfloat16 *>(out->data()), 
        reinterpret_cast<const __nv_bfloat16 *>(in->data()), 
        reinterpret_cast<const int64_t *>(pos_ids->data()), theta, seqlen, nhead, d);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia