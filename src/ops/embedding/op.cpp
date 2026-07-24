#include "op.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llaisys::ops {

template <typename T>
void embedding_(T* out, const int64_t* index, const T* weight, 
    size_t n, size_t d, size_t weight_stride0) {
        for (size_t i = 0; i < n; i++) {
            int64_t row = index[i];
            const T* src = weight + row * weight_stride0;
            T* dst = out + i * d;
            memcpy(dst, src, d * sizeof(T));
        }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    size_t n = index->numel();
    size_t d = weight->shape()[1];
    size_t weight_s0 = weight->strides()[0];
    auto dtype = weight->dtype();

    ASSERT(out->shape()[0] == n, "embedding out dim0 mismatch");
    ASSERT(out->shape()[1] == d, "");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "");

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float*>(out->data()), 
            reinterpret_cast<const int64_t*>(index->data()), 
            reinterpret_cast<const float*>(weight->data()), n, d, weight_s0);
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<bf16_t*>(out->data()), 
            reinterpret_cast<const int64_t*>(index->data()), 
            reinterpret_cast<const bf16_t*>(weight->data()), n, d, weight_s0);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<fp16_t*>(out->data()), 
            reinterpret_cast<const int64_t*>(index->data()), 
            reinterpret_cast<const fp16_t*>(weight->data()), n, d, weight_s0);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
