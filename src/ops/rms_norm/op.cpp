#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template <typename T>
void rms_norm_(T* out, const T* in, const T* weight, float eps, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float val = utils::cast<float>(in[i * n + j]);
            sum_sq += val * val;
        }
        sum_sq = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
        for (size_t j = 0; j < n; j++) {
            out[i * n + j] = utils::cast<T>(
                utils::cast<float>(in[i * n + j]) * 
            utils::cast<float>(weight[j]) * 
            sum_sq);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t m = out->shape()[0];
    size_t n = out->shape()[1];
    auto dtype = out->dtype();
    switch(dtype) {
        case LLAISYS_DTYPE_BF16:
            return rms_norm_(
                reinterpret_cast<bf16_t*>(out->data()), 
                reinterpret_cast<const bf16_t*>(in->data()),
                reinterpret_cast<const bf16_t*>(weight->data()), eps, m, n
                );
        case LLAISYS_DTYPE_F16:
            return rms_norm_(
                reinterpret_cast<fp16_t*>(out->data()), 
                reinterpret_cast<const fp16_t*>(in->data()), 
                reinterpret_cast<const fp16_t*>(weight->data()), eps, m, n
                );
        case LLAISYS_DTYPE_F32:
            return rms_norm_(
                reinterpret_cast<float*>(out->data()), 
                reinterpret_cast<const float*>(in->data()), 
                reinterpret_cast<const float*>(weight->data()), eps, m, n
                );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);

    }
}
} // namespace llaisys::ops
