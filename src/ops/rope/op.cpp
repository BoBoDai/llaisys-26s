#include "op.hpp"
#include <cstddef>

#include "cpu/rope.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out, in, pos_ids, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out, in, pos_ids, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
