#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight,
                                size_t n, size_t d, size_t weight_stride0) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t total = n * d;
    for (size_t flat = blockIdx.x * blockDim.x + threadIdx.x; flat < total; flat += stride) {
        size_t i = flat / d;
        size_t j = flat % d;
        size_t row = static_cast<size_t>(index[i]);
        out[flat] = weight[row * weight_stride0 + j];
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape()[1];
    size_t weight_stride0 = weight->strides()[0];
    const size_t total = num_indices * embedding_dim;
    dispatchFloatTypes(weight->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        embeddingKernel<<<gridSize(total), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(out->data()),
            reinterpret_cast<const int64_t *>(index->data()),
            reinterpret_cast<const T *>(weight->data()),
            num_indices, embedding_dim, weight_stride0);
    });
    checkLaunch("embeddingKernel");
}

} // namespace llaisys::ops::metax
