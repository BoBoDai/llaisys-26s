#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int *_device_ids;
    int _ndevice;
    int _device_id;
    LlaisysQwen2Weights *_weights;

    // KV cache — one key/value tensor per layer
    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    size_t _cache_length = 0;

    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

public:
    static Qwen2Model *create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2Model();

    void loadWeights(const char *name, void *data, size_t size);
    LlaisysQwen2Weights *weights();
    int64_t infer(const int64_t *token_ids, size_t ntoken);
    void destroyWeights();
};

} // namespace llaisys::models