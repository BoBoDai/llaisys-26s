#include "op.hpp"

#include "cpu/rms_norm.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rms_norm_metax.mch"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm only supports contiguous tensors");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out, in, weight, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out, in, weight, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out, in, weight, eps);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rms_norm(out, in, weight, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
