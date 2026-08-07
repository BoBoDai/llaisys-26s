#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        float g = toFloat(gate[i]);
        float u = toFloat(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = fromFloat<T>(u * silu);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t numel = out->numel();
    dispatchFloatTypes(out->dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        swigluKernel<<<gridSize(numel), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(out->data()), reinterpret_cast<const T *>(gate->data()),
            reinterpret_cast<const T *>(up->data()), numel);
    });
    checkLaunch("swigluKernel");
}

} // namespace llaisys::ops::metax
