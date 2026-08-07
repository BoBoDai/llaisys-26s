#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.mch"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out, in, weight, bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out, in, weight, bias);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::linear(out, in, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
}