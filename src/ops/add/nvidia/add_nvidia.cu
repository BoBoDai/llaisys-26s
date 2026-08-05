#include "add_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <type_traits>

namespace llaisys::ops::nvidia {

// ---- device helpers (self-contained) ----
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

// ---- kernel ----
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        c[i] = _from_f32<T>(_to_f32(a[i]) + _to_f32(b[i]));
    }
}

// ---- host launcher ----
template <typename T>
void add_launch(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    const int block = 256;
    int grid = (int)((numel + block - 1) / block);
    if (grid > 65535) grid = 65535;
    if (grid < 1) grid = 1;
    add_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(c),
                                    reinterpret_cast<const T *>(a),
                                    reinterpret_cast<const T *>(b), numel);
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type,
         size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:  return add_launch<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:  return add_launch<__half>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16: return add_launch<__nv_bfloat16>(c, a, b, numel);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia