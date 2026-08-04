#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_SHAPE(out->shape(), index->shape(), weight->shape());
    CHECK_SAME_DTYPE(out->dtype(), index->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out, index, weight);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out, index, weight);
#ifdef LLAISYS_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
