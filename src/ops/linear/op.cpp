#include "op.hpp"

namespace llaisys::ops {

template <typename T>
void linear_(T* out, const T* in, const T* weight, const T* bias, size_t out_m, size_t out_n, size_t weight_k) {
    for (int i = 0; i < out_m; ++i) {
        for (int j = 0; j < out_n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < weight_k; ++k) {
                sum += utils::cast<float>(in[i * weight_k + k]) * 
                utils::cast<float>(weight[j * weight_k + k]);
            }
            if (bias != nullptr) {
                sum += utils::cast<float>(bias[j]);
            }
            out[i * out_n + j] = utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t out_m = out->shape()[0];
    size_t out_n = out->shape()[1];
    size_t weight_k = weight->shape()[1];

    auto dtype = out->dtype();
    switch(dtype) {
        case LLAISYS_DTYPE_BF16:
            return linear_(
                reinterpret_cast<bf16_t*>(out->data()), 
                reinterpret_cast<const bf16_t*>(in->data()),
                 reinterpret_cast<const bf16_t*>(weight->data()),
                 reinterpret_cast<const bf16_t*>(bias->data()), out_m, out_n, weight_k
                );
        case LLAISYS_DTYPE_F16:
            return linear_(
                reinterpret_cast<fp16_t*>(out->data()), 
                reinterpret_cast<const fp16_t*>(in->data()), 
                reinterpret_cast<const fp16_t*>(weight->data()),
                reinterpret_cast<const fp16_t*>(bias->data()), out_m, out_n, weight_k
                );
        case LLAISYS_DTYPE_F32:
            return linear_(
                reinterpret_cast<float*>(out->data()), 
                reinterpret_cast< const float*>(in->data()), 
                reinterpret_cast<const float*>(weight->data()),
                reinterpret_cast<const float*>(bias->data()), out_m, out_n, weight_k
                );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
