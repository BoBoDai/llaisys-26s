#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/swiglu_metax.mch"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out, gate, up);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out, gate, up);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out, gate, up);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::swiglu(out, gate, up);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops