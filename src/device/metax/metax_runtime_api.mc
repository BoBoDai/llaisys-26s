#include "../runtime_api.hpp"

#include <mc_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::metax {

namespace runtime_api {
namespace {
void checkMaca(mcError_t status, const char *operation) {
    if (status != mcSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + mcGetErrorString(status));
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    const mcError_t status = mcGetDeviceCount(&count);
    if (status != mcSuccess) {
        mcGetLastError();
        return 0;
    }
    return count;
}

void setDevice(int device) {
    checkMaca(mcSetDevice(device), "mcSetDevice");
}

void deviceSynchronize() {
    checkMaca(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    checkMaca(mcStreamCreate(&stream), "mcStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkMaca(mcStreamDestroy(reinterpret_cast<mcStream_t>(stream)), "mcStreamDestroy");
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkMaca(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)), "mcStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkMaca(mcMalloc(&ptr, size), "mcMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFree(ptr), "mcFree");
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkMaca(mcMallocHost(&ptr, size), "mcMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFreeHost(ptr), "mcFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t /*kind*/) {
    checkMaca(mcMemcpy(dst, src, size, mcMemcpyDefault), "mcMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t /*kind*/, llaisysStream_t stream) {
    checkMaca(mcMemcpyAsync(dst, src, size, mcMemcpyDefault, reinterpret_cast<mcStream_t>(stream)), "mcMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
