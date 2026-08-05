#include "embedding_nvidia.cuh"

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
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                  size_t n, size_t d, size_t weight_stride0) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t total = n * d;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t i = flat / d;
        size_t j = flat % d;
        size_t row = (size_t)index[i];
        out[flat] = weight[row * weight_stride0 + j];
    }
}

template <typename T>
void embedding_launch(T *out, const int64_t *index, const T *weight,
                       size_t n, size_t d, size_t weight_stride0) {
    const int block = 256;
    int grid = (int)(((n * d) + block - 1) / block);
    if (grid > 65535) grid = 65535;
    if (grid < 1) grid = 1;
    embedding_kernel<T><<<grid, block>>>(out, index, weight, n, d, weight_stride0);
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    size_t n = index->numel();
    size_t d = weight->shape()[1];
    size_t weight_s0 = weight->strides()[0];
    auto dtype = weight->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  
    return embedding_launch(
        reinterpret_cast<float *>(out->data()), 
        reinterpret_cast<const int64_t *>(index->data()), 
        reinterpret_cast<const float *>(weight->data()), n, d, weight_s0);
    case LLAISYS_DTYPE_F16:  
    return embedding_launch(
        reinterpret_cast<__half *>(out->data()), 
        reinterpret_cast<const int64_t *>(index->data()), 
        reinterpret_cast<const __half *>(weight->data()), n, d, weight_s0);
    case LLAISYS_DTYPE_BF16: 
    return embedding_launch(
        reinterpret_cast<__nv_bfloat16 *>(out->data()), 
        reinterpret_cast<const int64_t *>(index->data()), 
        reinterpret_cast<const __nv_bfloat16 *>(weight->data()), n, d, weight_s0);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia