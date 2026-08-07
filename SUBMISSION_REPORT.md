# LLAISYS 提交报告

## 一、项目概述

本项目基于 LLM 推理框架，实现了张量系统、8 个算子的 CPU/CUDA 双后端、Qwen2 模型端到端推理，关键改动如下。

### 1.1 主要改动

| 模块 | 改动 |
| :--- | :--- |
| 张量系统 | 实现 load、isContiguous、view、permute、slice，storage + offset + meta 分离，view/permute/slice 零拷贝 |
| 算子层 | 实现 add、argmax、embedding、linear、rms_norm、rope、self_attention、swiglu 共 8 个算子，支持 Float32/Float16/BFloat16，CPU/CUDA 双后端 |
| 模型推理 | 实现 Qwen2 完整推理管线（embedding → N×transformer block → lm_head → argmax），含 KV cache |
| CUDA 加速 | 全部 8 个算子的 CUDA kernel，linear 接入 cuBLAS，self_attention 实现 online softmax |
| 设备抽象 | RuntimeAPI 函数指针表（malloc/free/memcpy/stream），编译期宏切换 CPU/NVIDIA 后端，上层模型代码不感知设备 |

---

## 二、平台支持

### 2.1 支持的平台及测试状态

| 平台 | 运行时 | 算子 | Qwen2 推理 | 状态 |
| :--- | :---: | :---: | :---: | :--- |
| CPU x86-64 (Linux, GCC) | Pass | Pass | Pass | 正式支持，CI 覆盖 |
| CPU x86-64 (Windows, MSVC) | Pass | Pass | Pass | 正式支持，CI 覆盖 |
| CPU ARM (Linux) | Pass | Pass | Pass | 本地验证通过，CI 未覆盖 |
| NVIDIA GPU (CUDA) | Pass | Pass | Pass | 本地验证通过（RTX 5090 ×1, 32 GB），CI 未覆盖 |

状态图例：Pass 全部测试通过 | — 理论兼容但未在 CI 验证 | Fail 不支持

### 2.2 平台说明

#### CPU x86-64 (Linux / Windows)

CI 在 `ubuntu-latest` 和 `windows-latest` 双平台运行。Windows 使用 MSVC 2022 编译器，Linux 使用 GCC。本项目 CPU 实现为纯 C++17 无 SIMD intrinsic 无 OpenMP，因此跨编译器兼容性良好。如果后续引入 SIMD 或 OpenMP，需注意 MSVC 仅支持 OpenMP 2.0（不支持 `collapse` 子句）。

#### NVIDIA GPU

全部 8 个算子 + 模型推理在 CUDA 后端通过测试。编译需 `xmake f --nv-gpu=y` 启用 CUDA 支持。依赖 CUDA Toolkit 和 cuBLAS。

---

## 三、构建与复现

### 3.1 环境要求

| 依赖 | 版本/说明 |
| :--- | :--- |
| Xmake | latest（CI 使用 `xmake-io/github-action-setup-xmake@v1`） |
| C++ 编译器 | MSVC 2022 (Windows) / GCC 或 Clang (Linux)，需支持 C++17 |
| Python | >= 3.9，需 PyTorch、Transformers、HuggingFace Hub |
| CUDA Toolkit（可选） | 仅 NVIDIA 后端需要 |

### 3.2 复现步骤

```bash
# 1. 克隆仓库
git clone <repo-url>
cd llaisys

# 2. CPU 构建与测试
xmake
xmake install
pip install ./python/

# 验证算子
python test/test_runtime.py --device cpu
python test/test_tensor.py --device cpu
python test/ops/add.py --device cpu
python test/ops/argmax.py --device cpu
python test/ops/embedding.py --device cpu
python test/ops/linear.py --device cpu
python test/ops/rms_norm.py --device cpu
python test/ops/rope.py --device cpu
python test/ops/self_attention.py --device cpu
python test/ops/swiglu.py --device cpu

# 端到端推理（需先下载模型）
python test/test_infer.py --model <path/to/DeepSeek-R1-Distill-Qwen-1.5B> --test

# 3. CUDA 构建与测试（需 GPU）
xmake f --nv-gpu=y -c
xmake
xmake install

# 算子测试
python test/ops/add.py --device nvidia
# ... 其余算子同理，加 --device nvidia

# 端到端推理
python test/test_infer.py --model <path/to/model> --test --device nvidia
```

### 3.3 复现结果

所有测试在以下环境验证通过：

#### Linux (Ubuntu, GCC)

```text
test_runtime.py --device cpu         PASS
test_tensor.py                       PASS
test/ops/*.py (8 个算子, 3 种精度)    PASS
test_infer.py --test                 PASS  128 token 与 HuggingFace 输出完全一致
```

#### Windows (MSVC 2022)

```text
（CI 验证，同上全绿）
```

#### NVIDIA RTX（RTX 5090 ×1, 32 GB）

```text
test/ops/*.py --device nvidia        PASS  全部通过
test_infer.py --test --device nvidia PASS  128 token 完全一致
```

---

## 四、性能数据

### 4.1 端到端推理（`test/test_infer.py --test`）

DeepSeek-R1-Distill-Qwen-1.5B，prompt="Who are you?"，生成 ~85 token，确定性采样（temp=1.0, top_k=1）。token 序列与 HuggingFace Transformers 完全一致，正确性已验证。

| 设备 | 总耗时 | ms/token | 输出正确 |
| :--- | :---: | :---: | :---: |
| CPU | 445.08s | ~5,236 | PASS |
| GPU | **0.49s** | **~5.8** | PASS |

### 4.2 单算子 Micro-benchmark（Qwen2 1.5B decode 形状，f32）

| 算子 | CPU (ms) | GPU (ms) | 加速比 |
| :--- | :---: | :---: | :---: |
| linear (head) | 192.31 | 0.63 | 304× |
| linear (down) | 10.33 | 0.015 | 698× |
| linear (gate/up) | 9.96 | 0.020 | 511× |
| linear (proj) | 1.86 | 0.008 | 245× |
| self_attention | 0.28 | 0.040 | 6.8× |
| argmax | 0.083 | 0.014 | 5.9× |
| swiglu | 0.024 | 0.003 | 7.9× |
| rope (q) | 0.007 | 0.003 | 2.1× |
| rms_norm | 0.002 | 0.003 | 0.7× |
| add | 0.0007 | 0.003 | 0.2× |
| **合计** | **214.86** | **0.75** | **288×** |
