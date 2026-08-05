#include "self_attention_nvidia.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
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

// ---------- kernel 1: scores = Q @ K^T * scale ----------
// scores are always float (allocated via cudaMalloc), to avoid precision issues in softmax.
template <typename T>
__global__ void attn_qk_kernel(float *scores, const T *q, const T *k,
                                size_t seqlen, size_t nhead, size_t d,
                                size_t total_len, size_t nkvhead, float scale) {
    size_t total = seqlen * nhead * total_len;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t j = flat % total_len;
        size_t rem = flat / total_len;
        size_t h = rem % nhead;
        size_t i = rem / nhead;
        size_t hk = h * nkvhead / nhead;

        float dot = 0.0f;
        for (size_t kk = 0; kk < d; ++kk)
            dot += _to_f32(q[i * nhead * d + h * d + kk]) *
                   _to_f32(k[j * nkvhead * d + hk * d + kk]);
        scores[flat] = dot * scale;
    }
}

// ---------- kernel 2: causal mask + softmax (in-place on float scores) ----------
__global__ void attn_softmax_kernel(float *scores, size_t seqlen, size_t nhead,
                                     size_t total_len) {
    size_t past_len = total_len - seqlen;
    size_t total_rows = seqlen * nhead;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total_rows; flat += stride) {
        size_t h = flat % nhead;
        size_t i = flat / nhead;
        size_t off = (i * nhead + h) * total_len;
        size_t limit = past_len + i;

        // causal mask
        for (size_t j = 0; j < total_len; ++j) {
            if (j > limit) scores[off + j] = -INFINITY;
        }

        // max
        float m = scores[off];
        for (size_t j = 1; j < total_len; ++j)
            if (scores[off + j] > m) m = scores[off + j];

        // exp + sum
        float sum = 0.0f;
        for (size_t j = 0; j < total_len; ++j) {
            scores[off + j] = expf(scores[off + j] - m);
            sum += scores[off + j];
        }
        if (sum > 0.0f)
            for (size_t j = 0; j < total_len; ++j) scores[off + j] /= sum;
        else
            for (size_t j = 0; j < total_len; ++j) scores[off + j] = 0.0f;
    }
}

// ---------- kernel 3: out = softmax(scores) @ V ----------
template <typename T>
__global__ void attn_out_kernel(T *out, const float *scores, const T *v,
                                 size_t seqlen, size_t nhead, size_t total_len,
                                 size_t nkvhead, size_t dv) {
    size_t total = seqlen * nhead * dv;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t m = flat % dv;
        size_t rem = flat / dv;
        size_t h = rem % nhead;
        size_t i = rem / nhead;
        size_t hk = h * nkvhead / nhead;

        float sum = 0.0f;
        size_t soff = (i * nhead + h) * total_len;
        for (size_t j = 0; j < total_len; ++j)
            sum += scores[soff + j] * _to_f32(v[j * nkvhead * dv + hk * dv + m]);
        out[flat] = _from_f32<T>(sum);
    }
}

// ---------- host launcher ----------
template <typename T>
void self_attn_launch(T *attn_val, const T *q, const T *k, const T *v, float scale,
                       size_t seqlen, size_t nhead, size_t d,
                       size_t total_len, size_t nkvhead, size_t dv) {
    const int block = 256;
    size_t n_scores = seqlen * nhead * total_len;
    float *scores;
    cudaMalloc(&scores, n_scores * sizeof(float));

    // kernel 1: QK^T
    {
        int grid = (int)((n_scores + block - 1) / block);
        if (grid > 65535) grid = 65535;
        if (grid < 1) grid = 1;
        attn_qk_kernel<T><<<grid, block>>>(scores, q, k, seqlen, nhead, d,
                                            total_len, nkvhead, scale);
    }
    // kernel 2: causal softmax
    {
        size_t n_rows = seqlen * nhead;
        int grid = (int)((n_rows + block - 1) / block);
        if (grid > 65535) grid = 65535;
        if (grid < 1) grid = 1;
        attn_softmax_kernel<<<grid, block>>>(scores, seqlen, nhead, total_len);
    }
    // kernel 3: scores @ V
    {
        size_t n_out = seqlen * nhead * dv;
        int grid = (int)((n_out + block - 1) / block);
        if (grid > 65535) grid = 65535;
        if (grid < 1) grid = 1;
        attn_out_kernel<T><<<grid, block>>>(attn_val, scores, v,
                                             seqlen, nhead, total_len, nkvhead, dv);
    }
    cudaFree(scores);
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    auto dtype = attn_val->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attn_launch(
            reinterpret_cast<float *>(attn_val->data()),
            reinterpret_cast<const float *>(q->data()), 
            reinterpret_cast<const float *>(k->data()), 
            reinterpret_cast<const float *>(v->data()), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attn_launch(
            reinterpret_cast<__half *>(attn_val->data()), 
            reinterpret_cast<const __half *>(q->data()), 
            reinterpret_cast<const __half *>(k->data()), 
            reinterpret_cast<const __half *>(v->data()), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attn_launch(
            reinterpret_cast<__nv_bfloat16 *>(attn_val->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(q->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(k->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(v->data()), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia