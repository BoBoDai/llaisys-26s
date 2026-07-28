#include "op.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace llaisys::ops {

void causalsoftmax(float* scores, size_t seqlen, size_t nhead, size_t total_len) {
    // const float NEG_INF = -1e9f;
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    size_t past_len = total_len - seqlen;

    for (size_t i = 0; i < seqlen; i++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t row_offset = ((i * nhead) + h) * total_len;

            size_t visible_limit = past_len + i;

            for (size_t j = visible_limit + 1; j < total_len; j++) {
                scores[row_offset + j] = NEG_INF;

            }

            float max_val = scores[row_offset];
            for (size_t j = 1; j <= visible_limit; j++) {
                if (scores[row_offset + j] > max_val) {
                    max_val = scores[row_offset + j];  
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                float val = std::exp(scores[row_offset + j] - max_val);
                scores[row_offset + j] = val;
                sum_exp += val;
            }

            if (sum_exp > 0.0f) {
                for (size_t j = 0; j < total_len; j++) {
                    scores[row_offset + j] /= sum_exp;
                }
            } else {
                for (size_t j = 0; j < total_len; j++) {
                    scores[row_offset + j] = 0.0f;
                }
            }
        }

    }

}

template <typename T>
void self_attention_(T* attn_val, const T* q, const T* k, const T* v, 
    float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {

        std::vector<float> scores(seqlen * nhead * total_len, 0.0f);

        for (size_t i = 0; i < seqlen; i++) {
            for (size_t h = 0; h < nhead; h++) {
                for (size_t j = 0; j < total_len; j++) {
                    float sum = 0;
                    for (size_t k_idx = 0; k_idx < d; k_idx++) {
                        sum += utils::cast<float>(q[i * nhead * d + h * d + k_idx]) *
                               utils::cast<float>(k[j * nkvhead * d + (h * nkvhead / nhead) * d + k_idx]);
                    }
                    scores[i * nhead * total_len + h * total_len + j] = sum * scale;
                }
            }
        }

        causalsoftmax(scores.data(), seqlen, nhead, total_len);

        for (size_t i = 0; i < seqlen; i++) {
            for (size_t h = 0; h < nhead; h++) {
                size_t h_k = h * nkvhead / nhead;
                size_t scores_offset = i * nhead * total_len + h * total_len;
                size_t out_offset = i * nhead * dv + h * dv;

                for (size_t m = 0; m < dv; m++) {
                    float weighted_sum = 0.0f;

                    for (size_t j = 0; j < total_len; j++) {
                        float prob = scores[scores_offset + j];
                        
                        size_t v_idx = j * nkvhead * dv + h_k * dv + m;
                        weighted_sum += prob * utils::cast<float>(v[v_idx]);
                    }
                    attn_val[out_offset + m] = utils::cast<T>(weighted_sum);
                }
            }
        }




}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    auto dtype = attn_val->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_BF16:
            return self_attention_(
                reinterpret_cast<bf16_t*>(attn_val->data()), 
                reinterpret_cast<const bf16_t*>(q->data()),
                reinterpret_cast<const bf16_t*>(k->data()),
                reinterpret_cast<const bf16_t*>(v->data()),
                scale, seqlen, nhead, d, total_len, nkvhead, dv);
        case LLAISYS_DTYPE_F16:
            return self_attention_(
                reinterpret_cast<fp16_t*>(attn_val->data()), 
                reinterpret_cast<const fp16_t*>(q->data()),
                reinterpret_cast<const fp16_t*>(k->data()),
                reinterpret_cast<const fp16_t*>(v->data()),
                scale, seqlen, nhead, d, total_len, nkvhead, dv);
        case LLAISYS_DTYPE_F32:
            return self_attention_(
                reinterpret_cast<float*>(attn_val->data()), 
                reinterpret_cast<const float*>(q->data()),
                reinterpret_cast<const float*>(k->data()),
                reinterpret_cast<const float*>(v->data()),
                scale, seqlen, nhead, d, total_len, nkvhead, dv);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
