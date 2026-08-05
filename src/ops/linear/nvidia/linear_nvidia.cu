#include "linear_nvidia.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <type_traits>

namespace llaisys::ops::nvidia {

// ---- cuBLAS handle (lazy init, one per device) ----
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

// ---- bias addition kernel ----
template <typename T>
__device__ __forceinline__ T _add_bias_elem(const T &v, const T &b) {
    if constexpr (std::is_same_v<T, float>) return v + b;
    else if constexpr (std::is_same_v<T, __half>) return __hadd(v, b);
    else return __hadd(v, b);
}

template <typename T>
__global__ void bias_kernel(T *out, const T *bias, size_t out_m, size_t out_n) {
    size_t row = blockIdx.x;
    if (row >= out_m) return;
    for (size_t col = threadIdx.x; col < out_n; col += blockDim.x) {
        out[row * out_n + col] = _add_bias_elem(out[row * out_n + col], bias[col]);
    }
}

// ---- dtype -> cublas type mapping ----
template <typename T> struct cublas_traits {};

template <> struct cublas_traits<float> {
    static constexpr cudaDataType_t type = CUDA_R_32F;
    static constexpr cudaDataType_t compute = CUDA_R_32F;
};

template <> struct cublas_traits<__half> {
    static constexpr cudaDataType_t type = CUDA_R_16F;
    static constexpr cudaDataType_t compute = CUDA_R_32F;
};

template <> struct cublas_traits<__nv_bfloat16> {
    static constexpr cudaDataType_t type = CUDA_R_16BF;
    static constexpr cudaDataType_t compute = CUDA_R_32F;
};

// ---- cuBLAS GEMM launcher ----
template <typename T>
void linear_gemm(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t M = out->shape()[0];   // batch / seqlen
    size_t N = out->shape()[1];   // output features
    size_t K = weight->shape()[1]; // input features

    T *out_ptr    = reinterpret_cast<T *>(out->data());
    T *in_ptr     = reinterpret_cast<T *>(in->data());
    T *weight_ptr = reinterpret_cast<T *>(weight->data());

    cublasHandle_t handle = get_cublas_handle();

    // out[M,N] = in[M,K] @ weight[N,K]^T
    // cuBLAS row-major: C[M,N] = A[M,K] * B[K,N] where B = weight^T
    // -> cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    //                 N, M, K, &alpha, B, type, ldb, A, type, lda, &beta, C, type, ldc, ...)
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 (int)N, (int)M, (int)K,
                 &alpha,
                 weight_ptr, cublas_traits<T>::type, (int)K,
                 in_ptr,     cublas_traits<T>::type, (int)K,
                 &beta,
                 out_ptr,    cublas_traits<T>::type, (int)N,
                 cublas_traits<T>::compute,
                 CUBLAS_GEMM_DEFAULT);

    // add bias if present
    if (bias && bias->data()) {
        T *bias_ptr = reinterpret_cast<T *>(bias->data());
        bias_kernel<T><<<(unsigned)M, 256>>>(out_ptr, bias_ptr, M, N);
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto dtype = out->dtype();
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  return linear_gemm<float>(out, in, weight, bias);
    case LLAISYS_DTYPE_F16:  return linear_gemm<__half>(out, in, weight, bias);
    case LLAISYS_DTYPE_BF16: return linear_gemm<__nv_bfloat16>(out, in, weight, bias);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
