#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void attnQkKernel(float *scores, const T *q, const T *k,
                             size_t seqlen, size_t nhead, size_t d,
                             size_t total_len, size_t nkvhead, float scale) {
    size_t total = seqlen * nhead * total_len;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t j = flat % total_len;
        size_t rem = flat / total_len;
        size_t h = rem % nhead;
        size_t i = rem / nhead;
        size_t hk = h * nkvhead / nhead;

        float dot = 0.0f;
        for (size_t kk = 0; kk < d; ++kk) {
            dot += toFloat(q[i * nhead * d + h * d + kk]) *
                   toFloat(k[j * nkvhead * d + hk * d + kk]);
        }
        scores[flat] = dot * scale;
    }
}

__global__ void attnSoftmaxKernel(float *scores, size_t seqlen, size_t nhead,
                                  size_t total_len) {
    size_t past_len = total_len - seqlen;
    size_t total_rows = seqlen * nhead;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total_rows; flat += stride) {
        size_t h = flat % nhead;
        size_t i = flat / nhead;
        size_t off = (i * nhead + h) * total_len;
        size_t limit = past_len + i;

        // causal mask
        for (size_t j = 0; j < total_len; ++j) {
            if (j > limit) scores[off + j] = -macaInf();
        }

        // max
        float m = scores[off];
        for (size_t j = 1; j < total_len; ++j) {
            if (scores[off + j] > m) m = scores[off + j];
        }

        // exp + sum
        float sum = 0.0f;
        for (size_t j = 0; j < total_len; ++j) {
            scores[off + j] = expf(scores[off + j] - m);
            sum += scores[off + j];
        }
        if (sum > 0.0f) {
            for (size_t j = 0; j < total_len; ++j) scores[off + j] /= sum;
        } else {
            for (size_t j = 0; j < total_len; ++j) scores[off + j] = 0.0f;
        }
    }
}

template <typename T>
__global__ void attnOutKernel(T *out, const float *scores, const T *v,
                              size_t seqlen, size_t nhead, size_t total_len,
                              size_t nkvhead, size_t dv) {
    size_t total = seqlen * nhead * dv;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t m = flat % dv;
        size_t rem = flat / dv;
        size_t h = rem % nhead;
        size_t i = rem / nhead;
        size_t hk = h * nkvhead / nhead;

        float sum = 0.0f;
        size_t soff = (i * nhead + h) * total_len;
        for (size_t j = 0; j < total_len; ++j) {
            sum += scores[soff + j] * toFloat(v[j * nkvhead * dv + hk * dv + m]);
        }
        out[flat] = fromFloat<T>(sum);
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t q_len = q->shape()[0];
    size_t kv_len = k->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t num_kv_heads = k->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t n_scores = q_len * num_heads * kv_len;
    float *scores = nullptr;
    checkMaca(mcMalloc(reinterpret_cast<void **>(&scores), n_scores * sizeof(float)),
              "mcMalloc scores");

    dispatchFloatTypes(attn_val->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        attnQkKernel<<<gridSize(n_scores), BLOCK_SIZE>>>(
            scores, reinterpret_cast<const T *>(q->data()),
            reinterpret_cast<const T *>(k->data()),
            q_len, num_heads, head_dim, kv_len, num_kv_heads, scale);
    });
    checkLaunch("attnQkKernel");

    size_t n_rows = q_len * num_heads;
    attnSoftmaxKernel<<<gridSize(n_rows), BLOCK_SIZE>>>(
        scores, q_len, num_heads, kv_len);
    checkLaunch("attnSoftmaxKernel");

    size_t n_out = q_len * num_heads * head_dim;
    dispatchFloatTypes(attn_val->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        attnOutKernel<<<gridSize(n_out), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(attn_val->data()), scores,
            reinterpret_cast<const T *>(v->data()),
            q_len, num_heads, kv_len, num_kv_heads, head_dim);
    });
    checkLaunch("attnOutKernel");
    checkMaca(mcFree(scores), "mcFree scores");
}

} // namespace llaisys::ops::metax
