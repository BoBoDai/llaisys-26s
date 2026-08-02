#include "llaisys/models/qwen2.h"
#include <iostream>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
};

__C {

    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
            std::cout << "Creating model with " << meta->nlayer << " layers" << std::endl;
        return NULL;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        std::cout << "Destroying model" << std::endl;
        return;
    }

    void *llaisysQwen2ModelLoadWeights(LlaisysQwen2Model * model, const char *name, void *data, size_t size) {
        std::cout << "Loading weight: " << name << " with size: " << size << std::endl;
        return NULL;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        return NULL;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken) {
        std::cout << "Running inference with " << ntoken << " tokens" << std::endl;
        return 0;
    }
}