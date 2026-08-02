#pragma once
#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"

__C {
    typedef struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *model;
    } LlaisysQwen2Model;
}
