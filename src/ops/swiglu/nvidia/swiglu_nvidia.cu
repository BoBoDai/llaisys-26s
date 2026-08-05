#include "swiglu_nvidia.cuh"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ __forceinline__ float _to_f32(const T &v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else {
        return __bfloat162float(v);
    }
}
template <typename T>
__device__ __forceinline__ T _from_f32(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        float g = _to_f32(gate[i]);
        float u = _to_f32(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = _from_f32<T>(u * silu);
    }
}

template <typename T>
void swiglu_launch(T *out, const T *gate, const T *up, size_t numel) {
    const int block = 256;
    int grid = (int)((numel + block - 1) / block);
    if (grid > 65535) {
        grid = 65535;
    }
    if (grid < 1) {
        grid = 1;
    }
    swiglu_kernel<T><<<grid, block>>>(out, gate, up, numel);
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t numel = out->numel();
    auto dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_launch(
            reinterpret_cast<float *>(out->data()),
         reinterpret_cast<const float *>(gate->data()), 
         reinterpret_cast<const float *>(up->data()), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_launch(
            reinterpret_cast<__half *>(out->data()), 
            reinterpret_cast<const __half *>(gate->data()), 
            reinterpret_cast<const __half *>(up->data()), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_launch(
            reinterpret_cast<__nv_bfloat16 *>(out->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(gate->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(up->data()), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia