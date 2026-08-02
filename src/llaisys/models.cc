#include "llaisys_models.hpp"
#include <iostream>

__C {

    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
            return new LlaisysQwen2Model{llaisys::models::Qwen2Model::create(meta, device, device_ids, ndevice)};
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        std::cout << "Destroying model" << std::endl;
        delete model;
    }

    void llaisysQwen2ModelLoadWeights(LlaisysQwen2Model * model, const char *name, void *data, size_t size) {
        model->model->loadWeights(name, data, size);
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        return model->model->weights();
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}