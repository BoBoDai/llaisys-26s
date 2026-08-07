#include <cmath>

#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids,
                           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half_d = d / 2;
    size_t seq_stride = nhead * d;
    size_t total = seqlen * nhead * half_d;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t j = flat % half_d;
        size_t rem = flat / half_d;
        size_t h = rem % nhead;
        size_t i = rem / nhead;

        float pos = static_cast<float>(pos_ids[i]);
        // Use double precision for angle computation to reduce precision loss for large pos
        double angle = static_cast<double>(pos) / pow(static_cast<double>(theta),
                       2.0 * static_cast<double>(j) / static_cast<double>(d));
        float sin_value = static_cast<float>(sin(angle));
        float cos_value = static_cast<float>(cos(angle));

        size_t off = i * seq_stride + h * d;
        float a = toFloat(in[off + j]);
        float b = toFloat(in[off + j + half_d]);

        out[off + j] = fromFloat<T>(a * cos_value - b * sin_value);
        out[off + j + half_d] = fromFloat<T>(b * cos_value + a * sin_value);
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t seq_len = out->shape()[0];
    size_t num_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];
    size_t total = seq_len * num_heads * (head_dim / 2);
    dispatchFloatTypes(out->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        ropeKernel<<<gridSize(total), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(out->data()), reinterpret_cast<const T *>(in->data()),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            seq_len, num_heads, head_dim, theta);
    });
    checkLaunch("ropeKernel");
}

} // namespace llaisys::ops::metax
