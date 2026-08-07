#include "device/metax/metax_kernel_utils.mch"
#include "tensor/tensor.hpp"

namespace llaisys::ops::metax {

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t out_m, size_t out_n) {
    size_t row = blockIdx.x;
    if (row >= out_m) return;
    for (size_t col = threadIdx.x; col < out_n; col += blockDim.x) {
        out[row * out_n + col] = fromFloat<T>(toFloat(out[row * out_n + col]) + toFloat(bias[col]));
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    mcGetLastError(); // clear stale errors from cu-bridge runtime
    size_t m = out->shape()[0];
    size_t n = out->shape()[1];
    size_t k = weight->shape()[1];

    CHECK_ARGUMENT(m <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                       n <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                       k <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "Linear dimensions exceed mcBLAS limits.");

    mcblasHandle_t handle = mcblas_cache.get();
    const float alpha = 1.0f, beta = 0.0f;

    macaDataType_t data_type;
    mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_PEDANTIC;
    llaisysDataType_t dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        data_type = MACA_R_32F;
        break;
    case LLAISYS_DTYPE_F16:
        data_type = MACA_R_16F;
        compute_type = MCBLAS_COMPUTE_32F;
        break;
    case LLAISYS_DTYPE_BF16:
        data_type = MACA_R_16BF;
        compute_type = MCBLAS_COMPUTE_32F;
        break;
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    // out[M,N] = in[M,K] @ weight[N,K]^T
    // mcBLAS col-major (like cuBLAS): C = op(A) * op(B)
    // A = weight^T (N x K), B = in^T (K x M) → C = N x M
    // Equivalent to row-major: out[M,N] = in[M,K] @ weight[N,K]^T
    checkMcblas(mcblasGemmEx(handle, MCBLAS_OP_T, MCBLAS_OP_N,
                             static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                             &alpha, weight->data(), data_type, static_cast<int>(k),
                             in->data(), data_type, static_cast<int>(k),
                             &beta, out->data(), data_type, static_cast<int>(n),
                             compute_type, MCBLAS_GEMM_DEFAULT_TENSOR_OP),
                "mcblasGemmEx");

    if (bias) {
        dispatchFloatTypes(dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            addBiasKernel<<<static_cast<unsigned>(m), BLOCK_SIZE>>>(
                reinterpret_cast<T *>(out->data()), reinterpret_cast<const T *>(bias->data()), m, n);
        });
        checkLaunch("addBiasKernel");
    }
}

} // namespace llaisys::ops::metax
