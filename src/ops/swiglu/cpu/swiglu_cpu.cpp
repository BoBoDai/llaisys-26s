#include "swiglu_cpu.hpp"
#include <cmath>

namespace llaisys::ops::cpu {
template<typename T>
void swiglu_(T* out, const T* gate, const T* up, size_t seqlen, size_t intermediate_size) {
    for (size_t i = 0; i < seqlen; i++) {
        for (size_t j = 0; j < intermediate_size; j++) {
            size_t idx = i * intermediate_size + j;
            float gate_val = utils::cast<float>(gate[idx]);
            float up_val = utils::cast<float>(up[idx]);

            float silu_val;
            if (gate_val >= 0) {
                silu_val = gate_val / (1.0f + std::exp(-gate_val));
            } else {
                float exp_gate = std::exp(gate_val);
                silu_val = gate_val * exp_gate / (1.0f + exp_gate);
            }
            out[idx] = utils::cast<T>(up_val * silu_val);
        }
    }

}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t seqlen = gate->shape()[0];
    size_t intermediate_size = gate->shape()[1];
    auto dtype = out->dtype();

    switch(dtype) {
        case LLAISYS_DTYPE_BF16:
            return swiglu_(reinterpret_cast<bf16_t*>(out->data()), 
                reinterpret_cast<const bf16_t*>(gate->data()), 
                reinterpret_cast<const bf16_t*>(up->data()), seqlen, intermediate_size);
        case LLAISYS_DTYPE_F16:
            return swiglu_(reinterpret_cast<fp16_t*>(out->data()), 
                reinterpret_cast<const fp16_t*>(gate->data()), 
                reinterpret_cast<const fp16_t*>(up->data()), seqlen, intermediate_size);
        case LLAISYS_DTYPE_F32:
            return swiglu_(reinterpret_cast<float*>(out->data()), 
                reinterpret_cast<const float*>(gate->data()), 
                reinterpret_cast<const float*>(up->data()), seqlen, intermediate_size);
        default:
            throw std::runtime_error("Unsupported data type");
    }
}
} // namespace llaisys::ops::cpu
