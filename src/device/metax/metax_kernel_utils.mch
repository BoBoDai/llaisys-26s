#pragma once

// Shared GPU utilities for MetaX op kernels.
// Only included by .mc files (compiled with mxcc -x maca), NOT by regular C++ code.

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mc_runtime.h>
#include <mcblas/mcblas.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "llaisys.h"
#include "utils.hpp"

namespace llaisys::ops::metax {
namespace {

constexpr int BLOCK_SIZE = 256;

// ---- error checking ----
inline void checkMaca(mcError_t status, const char *operation) {
    if (status != mcSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + mcGetErrorString(status));
    }
}
inline void checkMcblas(mcblasStatus_t status, const char *operation) {
    if (status != MCBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with mcBLAS status " +
                                 std::to_string(status));
    }
}
inline void checkLaunch(const char *kernel) { checkMaca(mcGetLastError(), kernel); }

inline unsigned int gridSize(size_t numel) {
    const size_t blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_ARGUMENT(blocks <= std::numeric_limits<unsigned int>::max(),
                   "MACA launch grid is too large.");
    return static_cast<unsigned int>(blocks);
}

// ---- type conversion helpers (native MetaX) ----
template <typename T> __device__ float toFloat(T value);
template <> __device__ float toFloat<float>(float value) { return value; }
template <> __device__ float toFloat<__half>(__half value) { return __half2float(value); }
template <> __device__ float toFloat<__maca_bfloat16>(__maca_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ T fromFloat(float value);
template <> __device__ float fromFloat<float>(float value) { return value; }
template <> __device__ __half fromFloat<__half>(float value) { return __float2half_rn(value); }
template <> __device__ __maca_bfloat16 fromFloat<__maca_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ inline float macaInf() { return __int_as_float(0x7f800000); }

// ---- type dispatch ----
template <typename T> struct TypeTag { using type = T; };

template <typename Function>
void dispatchFloatTypes(llaisysDataType_t dtype, Function &&function) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  return function(TypeTag<float>{});
    case LLAISYS_DTYPE_F16:  return function(TypeTag<__half>{});
    case LLAISYS_DTYPE_BF16: return function(TypeTag<__maca_bfloat16>{});
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

// ---- mcBLAS handle cache ----
class McblasHandleCache {
private:
    mcblasHandle_t _handle = nullptr;
    int _device = -1;

public:
    ~McblasHandleCache() {
        if (_handle != nullptr) mcblasDestroy(_handle);
    }
    mcblasHandle_t get() {
        int device = 0;
        checkMaca(mcGetDevice(&device), "mcGetDevice");
        if (_handle == nullptr || _device != device) {
            if (_handle != nullptr) checkMcblas(mcblasDestroy(_handle), "mcblasDestroy");
            checkMcblas(mcblasCreate(&_handle), "mcblasCreate");
            _device = device;
        }
        checkMcblas(mcblasSetStream(_handle, nullptr), "mcblasSetStream");
        return _handle;
    }
};
static thread_local McblasHandleCache mcblas_cache;

} // namespace
} // namespace llaisys::ops::metax
