#include "model.hpp"
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace llaisys::models {
namespace {
tensor_t unwrap(llaisysTensor_t h) {
    return h ? h->tensor : nullptr;
}
} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : _meta(*meta), _device_type(device), _device_ids(device_ids), _ndevice(ndevice), _device_id(device_ids[0]) {
    CHECK_ARGUMENT(_meta.nlayer > 0 && _meta.hs > 0 && _meta.nh > 0 && _meta.nkvh > 0
                       && _meta.dh > 0 && _meta.di > 0 && _meta.maxseq > 0 && _meta.voc > 0,
                   "Qwen2 metadata dimensions must be positive.");
    CHECK_ARGUMENT(_meta.nh * _meta.dh == _meta.hs, "Qwen2: nh * dh must equal hs.");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0, "Qwen2: nh must be divisible by nkvh.");

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

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    auto &runtime = core::context().runtime();

    // Lazy KV cache allocation on first call
    if (_key_cache.empty()) {
        _key_cache.resize(_meta.nlayer);
        _value_cache.resize(_meta.nlayer);
        for (size_t l = 0; l < _meta.nlayer; l++) {
            _key_cache[l] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
            _value_cache[l] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        }
    }

    // Temp tensor helpers
    auto make_tensor = [&](const std::vector<size_t> &shape, llaisysDataType_t dtype) {
        return Tensor::create(shape, dtype, _device_type, _device_id);
    };
    auto make_fp = [&](const std::vector<size_t> &shape) {
        return make_tensor(shape, _meta.dtype);
    };

    // Position IDs
    auto indices = make_tensor({ntoken}, LLAISYS_DTYPE_I64);
    indices->load(token_ids);
    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; i++) positions[i] = static_cast<int64_t>(_cache_length + i);
    auto pos_ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64);
    pos_ids->load(positions.data());

    // Embedding
    auto hidden = make_fp({ntoken, _meta.hs});
    ops::embedding(hidden, indices, unwrap(_weights->in_embed));

    // Pre-allocated temp tensors (reused every layer)
    auto norm   = make_fp({ntoken, _meta.hs});
    auto q_flat = make_fp({ntoken, _meta.hs});
    size_t kv_dim = _meta.nkvh * _meta.dh;
    auto k_flat = make_fp({ntoken, kv_dim});
    auto v_flat = make_fp({ntoken, kv_dim});
    auto q_rot  = make_fp({ntoken, _meta.nh, _meta.dh});
    auto k_rot  = make_fp({ntoken, _meta.nkvh, _meta.dh});
    auto attn   = make_fp({ntoken, _meta.nh, _meta.dh});
    auto proj   = make_fp({ntoken, _meta.hs});
    auto gate   = make_fp({ntoken, _meta.di});
    auto up     = make_fp({ntoken, _meta.di});
    auto act    = make_fp({ntoken, _meta.di});
    auto down   = make_fp({ntoken, _meta.hs});

    size_t total_length = _cache_length + ntoken;
    size_t cache_bytes = ntoken * kv_dim * utils::dsize(_meta.dtype);

    for (size_t l = 0; l < _meta.nlayer; l++) {
        // RMS Norm + QKV projections
        ops::rms_norm(norm, hidden, unwrap(_weights->attn_norm_w[l]), _meta.epsilon);
        ops::linear(q_flat, norm, unwrap(_weights->attn_q_w[l]), unwrap(_weights->attn_q_b[l]));
        ops::linear(k_flat, norm, unwrap(_weights->attn_k_w[l]), unwrap(_weights->attn_k_b[l]));
        ops::linear(v_flat, norm, unwrap(_weights->attn_v_w[l]), unwrap(_weights->attn_v_b[l]));

        // Multi-head reshape + RoPE
        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});
        ops::rope(q_rot, q, pos_ids, _meta.theta);
        ops::rope(k_rot, k, pos_ids, _meta.theta);

        // Write to KV cache
        auto key_dst   = _key_cache[l]->slice(0, _cache_length, total_length);
        auto value_dst = _value_cache[l]->slice(0, _cache_length, total_length);
        runtime.api()->memcpy_async(key_dst->data(), k_rot->data(), cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());
        runtime.api()->memcpy_async(value_dst->data(), v->data(), cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());

        // Self attention over full context
        auto keys_all   = _key_cache[l]->slice(0, 0, total_length);
        auto values_all = _value_cache[l]->slice(0, 0, total_length);
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        ops::self_attention(attn, q_rot, keys_all, values_all, scale);

        // O projection + residual
        ops::linear(proj, attn->view({ntoken, _meta.hs}), unwrap(_weights->attn_o_w[l]), nullptr);
        ops::add(hidden, hidden, proj);

        // MLP: RMS Norm → gate/up → SwiGLU → down → residual
        ops::rms_norm(norm, hidden, unwrap(_weights->mlp_norm_w[l]), _meta.epsilon);
        ops::linear(gate, norm, unwrap(_weights->mlp_gate_w[l]), nullptr);
        ops::linear(up, norm, unwrap(_weights->mlp_up_w[l]), nullptr);
        ops::swiglu(act, gate, up);
        ops::linear(down, act, unwrap(_weights->mlp_down_w[l]), nullptr);
        ops::add(hidden, hidden, down);
    }

    _cache_length = total_length;

    // Final norm + lm_head + argmax
    ops::rms_norm(norm, hidden, unwrap(_weights->out_norm_w), _meta.epsilon);
    auto last = norm->slice(0, ntoken - 1, ntoken);
    auto logits = make_fp({1, _meta.voc});
    ops::linear(logits, last, unwrap(_weights->out_embed), nullptr);
    auto max_idx = make_tensor({1}, LLAISYS_DTYPE_I64);
    auto max_val = make_fp({1});
    ops::argmax(max_idx, max_val, logits->view({_meta.voc}));

    runtime.synchronize();
    int64_t result = 0;
    runtime.api()->memcpy_sync(&result, max_idx->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    return result;
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