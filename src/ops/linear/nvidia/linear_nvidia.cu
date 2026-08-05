#include "linear_nvidia.cuh"

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
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                               size_t out_m, size_t out_n, size_t weight_k) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_m || col >= out_n) return;

    float sum = 0.0f;
    for (size_t k = 0; k < weight_k; ++k) {
        sum += _to_f32(in[row * weight_k + k]) * _to_f32(weight[col * weight_k + k]);
    }
    if (bias != nullptr) sum += _to_f32(bias[col]);
    out[row * out_n + col] = _from_f32<T>(sum);
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t out_m = out->shape()[0];
    size_t out_n = out->shape()[1];
    size_t weight_k = weight->shape()[1];

    dim3 block(16, 16);
    dim3 grid((unsigned)((out_n + block.x - 1) / block.x),
              (unsigned)((out_m + block.y - 1) / block.y));

    auto dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const float *>(weight->data()),
            bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
            out_m, out_n, weight_k);
    case LLAISYS_DTYPE_F16:
        return linear_kernel<__half><<<grid, block>>>(
            reinterpret_cast<__half *>(out->data()),
            reinterpret_cast<const __half *>(in->data()),
            reinterpret_cast<const __half *>(weight->data()),
            bias ? reinterpret_cast<const __half *>(bias->data()) : nullptr,
            out_m, out_n, weight_k);
    case LLAISYS_DTYPE_BF16:
        return linear_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16 *>(out->data()),
            reinterpret_cast<const __nv_bfloat16 *>(in->data()),
            reinterpret_cast<const __nv_bfloat16 *>(weight->data()),
            bias ? reinterpret_cast<const __nv_bfloat16 *>(bias->data()) : nullptr,
            out_m, out_n, weight_k);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia