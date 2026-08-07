#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.mch"
#endif
#include <cassert>
#include <cstddef>
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out, index, weight);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out, index, weight);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out, index, weight);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
