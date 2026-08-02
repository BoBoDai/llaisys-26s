#include "model.hpp"
#include <iostream>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : _meta(*meta), _device_type(device), _device_ids(device_ids), _ndevice(ndevice) {
    _weights = new LlaisysQwen2Weights();
}

Qwen2Model::~Qwen2Model() {
    destroyWeights();
}

Qwen2Model *Qwen2Model::create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    std::cout << "Creating model with " << meta->nlayer << " layers" << std::endl;
    return new Qwen2Model(meta, device, device_ids, ndevice);
}

void Qwen2Model::loadWeights(const char *name, void *data, size_t size) {
    std::cout << "Loading weight: " << name << " with size: " << size << std::endl;
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return _weights;
}

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    std::cout << "Running inference with " << ntoken << " tokens" << std::endl;
    return 0;
}

void Qwen2Model::destroyWeights() {
    if (!_weights) return;
    delete[] _weights->attn_norm_w;
    delete[] _weights->attn_q_w;
    delete[] _weights->attn_q_b;
    delete[] _weights->attn_k_w;
    delete[] _weights->attn_k_b;
    delete[] _weights->attn_v_w;
    delete[] _weights->attn_v_b;
    delete[] _weights->attn_o_w;
    delete[] _weights->mlp_norm_w;
    delete[] _weights->mlp_gate_w;
    delete[] _weights->mlp_up_w;
    delete[] _weights->mlp_down_w;
    delete _weights;
    _weights = nullptr;
}

} // namespace llaisys::models