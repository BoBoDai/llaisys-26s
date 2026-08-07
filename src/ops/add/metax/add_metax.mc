#include "device/metax/metax_kernel_utils.mch"

namespace llaisys::ops::metax {

template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t numel) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        out[i] = fromFloat<T>(toFloat(a[i]) + toFloat(b[i]));
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype,
         size_t numel) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        addKernel<<<gridSize(numel), BLOCK_SIZE>>>(
            reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b), numel);
    });
    checkLaunch("addKernel");
}

} // namespace llaisys::ops::metax
