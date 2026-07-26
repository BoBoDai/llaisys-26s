#include "op.hpp"
#include <cstddef>
#include <cmath>

namespace llaisys::ops {
template <typename T>
void rope_(T* out, const T* in, const int* pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;
    size_t seq_stride = nhead * d;

    for (size_t i = 0; i < seqlen; i++) {
        float pos = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < nhead; h++) {
            size_t head_off = i * seq_stride + h * d;
            for (size_t j = 0; j < half_d; j++) {
                // Compute angle φ = pos / θ^(2j/d)
                float freq = pos / std::pow(theta,
                    2.0f * static_cast<float>(j) / static_cast<float>(d));
                float c = std::cos(freq);
                float s = std::sin(freq);

                float a = utils::cast<float>(in[head_off + j]);
                float b = utils::cast<float>(in[head_off + j + half_d]);

                out[head_off + j] = utils::cast<T>(a * c - b * s);
                out[head_off + j + half_d] = utils::cast<T>(b * c + a * s);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t m = out->shape()[0];
    size_t n = out->shape()[1];
    size_t k = out->shape()[2];
    auto dtype = out->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_BF16:
            return rope_(
                reinterpret_cast<bf16_t*>(out->data()),
                reinterpret_cast<const bf16_t*>(in->data()),
                reinterpret_cast<const int*>(pos_ids->data()), theta, m, n, k
            );
        case LLAISYS_DTYPE_F16:
            return rope_(
                reinterpret_cast<fp16_t*>(out->data()),
                reinterpret_cast<const fp16_t*>(in->data()),
                reinterpret_cast<const int*>(pos_ids->data()), theta, m, n, k
            );
        case LLAISYS_DTYPE_F32:
            return rope_(
                reinterpret_cast<float*>(out->data()),
                reinterpret_cast<const float*>(in->data()),
                reinterpret_cast<const int*>(pos_ids->data()), theta, m, n, k
            );
        default:\
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
}
} // namespace llaisys::ops
