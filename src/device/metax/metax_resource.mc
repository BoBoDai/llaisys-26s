#include "metax_resource.h"

namespace llaisys::device::metax {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_METAX, device_id) {}

Resource::~Resource() = default;

} // namespace llaisys::device::metax
