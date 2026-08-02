#include "llaisys/models/qwen2.h"

namespace llaisys::models {
class Qwen2Model;

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int *_device_ids;
    int _ndevice;
    LlaisysQwen2Weights *_weights;
    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

public:
    static Qwen2Model *create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2Model();

    void loadWeights(const char *name, void *data, size_t size);
    LlaisysQwen2Weights *weights();
    int64_t infer(int64_t *token_ids, size_t ntoken);
    void destroyWeights();
};

} // namespace llaisys::models