#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, float eps,
                              size_t m, size_t n) {
    size_t row = blockIdx.x;
    if (row >= m) return;

    float sum_sq = 0.0f;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        float v = toFloat(in[row * n + j]);
        sum_sq += v * v;
    }

    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(s)) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = 1.0f / sqrtf(s_sum[0] / static_cast<float>(n) + eps);

    __shared__ float s_scale;
    if (threadIdx.x == 0) s_scale = rms;
    __syncthreads();
    float scale = s_scale;

    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        out[row * n + j] = fromFloat<T>(toFloat(in[row * n + j]) * toFloat(weight[j]) * scale);
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t rows = out->shape()[0];
    size_t hidden = out->shape()[1];
    CHECK_ARGUMENT(rows <= std::numeric_limits<unsigned int>::max(),
                   "RMSNorm row count is too large.");
    dispatchFloatTypes(out->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        rmsNormKernel<<<static_cast<unsigned int>(rows), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(out->data()), reinterpret_cast<const T *>(in->data()),
            reinterpret_cast<const T *>(weight->data()), eps, rows, hidden);
    });
    checkLaunch("rmsNormKernel");
}

} // namespace llaisys::ops::metax
