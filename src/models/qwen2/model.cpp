#include "model.hpp"
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include <cstring>
#include <iostream>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : _meta(*meta), _device_type(device), _device_ids(device_ids), _ndevice(ndevice), _device_id(device_ids[0]) {
    _weights = new LlaisysQwen2Weights();
    size_t n = meta->nlayer;
    _weights->attn_norm_w = new llaisysTensor_t[n]();
    _weights->attn_q_w = new llaisysTensor_t[n]();
    _weights->attn_q_b = new llaisysTensor_t[n]();
    _weights->attn_k_w = new llaisysTensor_t[n]();
    _weights->attn_k_b = new llaisysTensor_t[n]();
    _weights->attn_v_w = new llaisysTensor_t[n]();
    _weights->attn_v_b = new llaisysTensor_t[n]();
    _weights->attn_o_w = new llaisysTensor_t[n]();
    _weights->mlp_norm_w = new llaisysTensor_t[n]();
    _weights->mlp_gate_w = new llaisysTensor_t[n]();
    _weights->mlp_up_w = new llaisysTensor_t[n]();
    _weights->mlp_down_w = new llaisysTensor_t[n]();
}

Qwen2Model::~Qwen2Model() {
    destroyWeights();
}

Qwen2Model *Qwen2Model::create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    return new Qwen2Model(meta, device, device_ids, ndevice);
}

void Qwen2Model::loadWeights(const char *name, void *data, size_t size) {
    int layer = -1;
    if (strncmp(name, "model.layers.", 13) == 0) {
        layer = atoi(name + 13);
    }

    std::vector<size_t> shape;
    llaisysTensor_t *slot = nullptr;
    if (strstr(name, "embed_tokens")) {
        shape = { _meta.voc, _meta.hs };
        slot = &_weights->in_embed;
    } else if (strstr(name, "lm_head")) {
        shape = { _meta.voc, _meta.hs };
        slot = &_weights->out_embed;
    } else if (strstr(name, "model.norm")) {
        shape = { _meta.hs };
        slot = &_weights->out_norm_w;
    }

    else if (strstr(name, "input_layernorm")) {
        shape = {_meta.hs}; slot = &_weights->attn_norm_w[layer];
    } else if (strstr(name, "q_proj.weight")) {
        shape = {_meta.nh * _meta.dh, _meta.hs}; slot = &_weights->attn_q_w[layer];
    } else if (strstr(name, "q_proj.bias")) {
        shape = {_meta.nh * _meta.dh}; slot = &_weights->attn_q_b[layer];
    } else if (strstr(name, "k_proj.weight")) {
        shape = {_meta.nkvh * _meta.dh, _meta.hs}; slot = &_weights->attn_k_w[layer];
    } else if (strstr(name, "k_proj.bias")) {
        shape = {_meta.nkvh * _meta.dh}; slot = &_weights->attn_k_b[layer];
    } else if (strstr(name, "v_proj.weight")) {
        shape = {_meta.nkvh * _meta.dh, _meta.hs}; slot = &_weights->attn_v_w[layer];
    } else if (strstr(name, "v_proj.bias")) {
        shape = {_meta.nkvh * _meta.dh}; slot = &_weights->attn_v_b[layer];
    } else if (strstr(name, "o_proj")) {
        shape = {_meta.hs, _meta.nh * _meta.dh}; slot = &_weights->attn_o_w[layer];
    } else if (strstr(name, "post_attention_layernorm")) {
        shape = {_meta.hs}; slot = &_weights->mlp_norm_w[layer];
    } else if (strstr(name, "gate_proj")) {
        shape = {_meta.di, _meta.hs}; slot = &_weights->mlp_gate_w[layer];
    } else if (strstr(name, "up_proj")) {
        shape = {_meta.di, _meta.hs}; slot = &_weights->mlp_up_w[layer];
    } else if (strstr(name, "down_proj")) {
        shape = {_meta.hs, _meta.di}; slot = &_weights->mlp_down_w[layer];
    } else {
        std::cerr << "Unknown weight: " << name << std::endl;
        return;
    }
    
    auto t = Tensor::create(shape, _meta.dtype, _device_type, _device_id);
    t->load(data);
    *slot = new LlaisysTensor{t};
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