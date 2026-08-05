#include "argmax_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
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

// Stage 1: each block reduces its portion to (max_val, min_index_on_tie).
// Results written to global partial arrays (float for value, int64 for index).
template <typename T>
__global__ void argmax_s1(const T *vals, size_t num, float *pval, int64_t *pidx) {
    __shared__ float sv[256];
    __shared__ int64_t si[256];

    float best = -INFINITY;
    int64_t idx = -1;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += stride) {
        float v = _to_f32(vals[i]);
        if (v > best)          { best = v; idx = (int64_t)i; }
        else if (v == best)    { /* keep smaller idx (first occurrence) */ }
    }

    unsigned tid = threadIdx.x;
    sv[tid] = best; si[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < (unsigned)s) {
            if (sv[tid + s] > sv[tid]) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            } else if (sv[tid + s] == sv[tid] && si[tid + s] >= 0 && si[tid + s] < si[tid]) {
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { pval[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}

// Stage 2: single block reduces partial arrays to final (idx, max_val)
__global__ void argmax_s2(const float *pval, const int64_t *pidx,
                           int64_t *o_idx, float *o_val, unsigned nblocks) {
    __shared__ float sv[256];
    __shared__ int64_t si[256];

    float best = -INFINITY;
    int64_t idx = -1;
    for (unsigned i = threadIdx.x; i < nblocks; i += blockDim.x) {
        if (pval[i] > best)           { best = pval[i]; idx = pidx[i]; }
        else if (pval[i] == best && pidx[i] >= 0 && pidx[i] < idx) { idx = pidx[i]; }
    }

    unsigned tid = threadIdx.x;
    sv[tid] = best; si[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < (unsigned)s) {
            if (sv[tid + s] > sv[tid]) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            } else if (sv[tid + s] == sv[tid] && si[tid + s] >= 0 && si[tid + s] < si[tid]) {
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { *o_idx = si[0]; *o_val = sv[0]; }
}

template <typename T>
void argmax_launch(T *max_val_ptr, const T *vals, int64_t *max_idx, size_t num) {
    const unsigned block = 256;
    unsigned nblocks = (unsigned)((num + block - 1) / block);
    if (nblocks > 65535) nblocks = 65535;
    if (nblocks < 1) nblocks = 1;

    float *pval;  int64_t *pidx;
    cudaMalloc(&pval, sizeof(float) * nblocks);
    cudaMalloc(&pidx, sizeof(int64_t) * nblocks);

    argmax_s1<T><<<nblocks, block>>>(vals, num, pval, pidx);
    argmax_s2<<<1, block>>>(pval, pidx, max_idx, reinterpret_cast<float *>(max_val_ptr), nblocks);

    cudaFree(pval); cudaFree(pidx);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx->data());
    size_t num = vals->numel();
    auto dtype = vals->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  
    return argmax_launch(
        reinterpret_cast<float *>(max_val->data()), 
        reinterpret_cast<const float *>(vals->data()), idx_ptr, num);
    case LLAISYS_DTYPE_F16:  
    return argmax_launch(
        reinterpret_cast<__half *>(max_val->data()), 
        reinterpret_cast<const __half *>(vals->data()), idx_ptr, num);
    case LLAISYS_DTYPE_BF16: 
    return argmax_launch(
        reinterpret_cast<__nv_bfloat16 *>(max_val->data()), 
        reinterpret_cast<const __nv_bfloat16 *>(vals->data()), idx_ptr, num);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia