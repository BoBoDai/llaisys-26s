#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void argmaxS1(const T *vals, size_t num, float *pval, int64_t *pidx) {
    __shared__ float sv[256];
    __shared__ int64_t si[256];

    float best = -macaInf();
    int64_t idx = -1;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += stride) {
        float v = toFloat(vals[i]);
        if (v > best) { best = v; idx = static_cast<int64_t>(i); }
    }
    unsigned tid = threadIdx.x;
    sv[tid] = best;
    si[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<unsigned>(s)) {
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

__global__ void argmaxS2(const float *pval, const int64_t *pidx,
                         int64_t *o_idx, float *o_val, unsigned nblocks) {
    __shared__ float sv[256];
    __shared__ int64_t si[256];

    float best = -macaInf();
    int64_t idx = -1;
    for (unsigned i = threadIdx.x; i < nblocks; i += blockDim.x) {
        if (pval[i] > best) { best = pval[i]; idx = pidx[i]; }
        else if (pval[i] == best && pidx[i] >= 0 && pidx[i] < idx) { idx = pidx[i]; }
    }
    unsigned tid = threadIdx.x;
    sv[tid] = best;
    si[tid] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<unsigned>(s)) {
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

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t numel = vals->numel();
    unsigned nblocks = gridSize(numel);
    float *pval = nullptr;
    int64_t *pidx = nullptr;
    checkMaca(mcMalloc(reinterpret_cast<void **>(&pval), sizeof(float) * nblocks), "mcMalloc pval");
    checkMaca(mcMalloc(reinterpret_cast<void **>(&pidx), sizeof(int64_t) * nblocks), "mcMalloc pidx");

    dispatchFloatTypes(vals->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        argmaxS1<<<nblocks, BLOCK_SIZE>>>(reinterpret_cast<const T *>(vals->data()), numel, pval, pidx);
    });
    checkLaunch("argmaxS1");
    argmaxS2<<<1, BLOCK_SIZE>>>(pval, pidx,
        reinterpret_cast<int64_t *>(max_idx->data()),
        reinterpret_cast<float *>(max_val->data()), nblocks);
    checkLaunch("argmaxS2");

    checkMaca(mcFree(pval), "mcFree pval");
    checkMaca(mcFree(pidx), "mcFree pidx");
}

} // namespace llaisys::ops::metax
