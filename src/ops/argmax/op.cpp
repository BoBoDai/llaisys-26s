#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void argmax_(int64_t* max_idx, T* max_val, const T* vals, size_t num) {
    *max_idx = 0;
    *max_val = vals[0];
    for (size_t i = 1; i < num; ++i) {
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            if (utils::cast<float>(vals[i]) > utils::cast<float>(*max_val)) {
                *max_val = vals[i];
                *max_idx = i;
            }
        } else {
            if (vals[i] > *max_val) {
                *max_val = vals[i];
                *max_idx = i;
            }
        }
    }
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    ASSERT(vals->ndim() == 1, "argmax only supports 1D tensors");
    size_t num = vals->numel();
    auto dtype = vals->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return argmax_(
                reinterpret_cast<int64_t*>(max_idx->data()),
                reinterpret_cast<float*>(max_val->data()),
                reinterpret_cast<const float*>(vals->data()),
                num
            );
        case LLAISYS_DTYPE_F64:
            return argmax_(
                reinterpret_cast<int64_t*>(max_idx->data()),
                reinterpret_cast<double*>(max_val->data()),
                reinterpret_cast<const double*>(vals->data()),
                num
            );
        case LLAISYS_DTYPE_BF16:
            return argmax_(
                reinterpret_cast<int64_t*>(max_idx->data()),
                reinterpret_cast<bf16_t*>(max_val->data()),
                reinterpret_cast<const bf16_t*>(vals->data()),
                num
            );
        case LLAISYS_DTYPE_F16:
            return argmax_(
                reinterpret_cast<int64_t*>(max_idx->data()),
                reinterpret_cast<fp16_t*>(max_val->data()),
                reinterpret_cast<const fp16_t*>(vals->data()),
                num
            );
        default:
            ASSERT(false, "Unsupported data type for argmax");
    }
}
} // namespace llaisys::ops
