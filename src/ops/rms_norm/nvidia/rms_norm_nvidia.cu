#include "rms_norm_nvidia.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
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
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps,
                                 size_t m, size_t n) {
    size_t row = blockIdx.x;
    if (row >= m) return;

    // per-thread partial sum of squares over this row
    float sum_sq = 0.0f;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        float v = _to_f32(in[row * n + j]);
        sum_sq += v * v;
    }

    // shared memory reduction (assumes blockDim <= 256)
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }

    float rms = 1.0f / sqrtf(s_sum[0] / (float)n + eps);

    // broadcast via shared memory, then write this row
    __shared__ float s_scale;
    if (threadIdx.x == 0) s_scale = rms;
    __syncthreads();
    float scale = s_scale;

    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        out[row * n + j] = _from_f32<T>(_to_f32(in[row * n + j]) * _to_f32(weight[j]) * scale);
    }
}

template <typename T>
void rms_norm_launch(T *out, const T *in, const T *weight, float eps, size_t m, size_t n) {
    const int block = 256;
    unsigned grid = (unsigned)m;
    if (grid < 1) grid = 1;
    rms_norm_kernel<T><<<grid, block>>>(out, in, weight, eps, m, n);
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t m = out->shape()[0];
    size_t n = out->shape()[1];
    auto dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  
    return rms_norm_launch(
        reinterpret_cast<float *>(out->data()), 
        reinterpret_cast<const float *>(in->data()), 
        reinterpret_cast<const float *>(weight->data()), eps, m, n);
    case LLAISYS_DTYPE_F16: 
     return rms_norm_launch(
        reinterpret_cast<__half *>(out->data()), 
        reinterpret_cast<const __half *>(in->data()), 
        reinterpret_cast<const __half *>(weight->data()), eps, m, n);
    case LLAISYS_DTYPE_BF16: 
    return rms_norm_launch(
        reinterpret_cast<__nv_bfloat16 *>(out->data()), 
        reinterpret_cast<const __nv_bfloat16 *>(in->data()), 
        reinterpret_cast<const __nv_bfloat16 *>(weight->data()), eps, m, n);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia