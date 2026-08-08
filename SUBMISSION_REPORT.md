# LLAISYS 提交报告

## 一、项目概述

本项目基于 LLM 推理框架，实现了张量系统、8 个算子的 CPU/CUDA/Metax 多后端、Qwen2 模型端到端推理，关键改动如下。

### 1.1 主要改动

| 模块       | 改动                                                                                                                         |
|:---------|:---------------------------------------------------------------------------------------------------------------------------|
| 张量系统     | 实现 load、isContiguous、view、permute、slice，storage + offset + meta 分离，view/permute/slice 零拷贝                                  |
| 算子层      | 实现 add、argmax、embedding、linear、rms_norm、rope、self_attention、swiglu 共 8 个算子，支持 Float32/Float16/BFloat16，CPU/NVIDIA/Metax 后端 |
| 模型推理     | 实现 Qwen2 完整推理管线（embedding → N×transformer block → lm_head → argmax），含 KV cache                                             |
| CUDA 加速  | 全部 8 个算子的 CUDA kernel                                                                                                      |
| Metax 加速 | 全部 8 个算子的 Metax kernel                                                                                                     |
| 设备抽象     | RuntimeAPI 函数指针表（malloc/free/memcpy/stream），编译期宏切换 CPU/NVIDIA/Metax 后端，上层模型代码不感知设备                                         |

---

## 二、平台支持

### 2.1 支持的平台及测试状态

| 平台                         | 运行时  |  算子  | Qwen2 推理 | 状态                             |
|:---------------------------|:----:|:----:|:--------:|:-------------------------------|
| CPU x86-64 (Linux, GCC)    | Pass | Pass |   Pass   | 正确支持，CI 覆盖                     |
| CPU x86-64 (Windows, MSVC) | Pass | Pass |   Pass   | 正确支持，CI 覆盖                     |
| CPU ARM (Linux)            | Pass | Pass |   Pass   | 本地验证通过                         |
| NVIDIA GPU (CUDA)          | Pass | Pass |   Pass   | 本地验证通过（RTX 4090 D - 24 GB × 1） |
| Metax GPU (MXC)            | Pass | Pass |   Pass   | 本地验证通过（曦云 C500 ×1, 16 GB）      |

### 2.2 平台说明

#### CPU x86-64 (Linux / Windows)

CI 在 `ubuntu-latest` 和 `windows-latest` 双平台运行。Windows 使用 MSVC 2022 编译器，Linux 使用 GCC。本项目 CPU 实现为纯
C++17 无 SIMD intrinsic 无 OpenMP，因此跨编译器兼容性良好。如果后续引入 SIMD 或 OpenMP，需注意 MSVC 仅支持 OpenMP 2.0（不支持
`collapse` 子句）。

#### NVIDIA GPU

全部 8 个算子 + 模型推理在 CUDA 后端通过测试。编译需 `xmake f --nv-gpu=y` 启用 CUDA 支持。依赖 CUDA Toolkit 和 cuBLAS。

#### Metax GPU

全部 8 个算子 + 模型推理在 Metax 后端通过测试。编译需 `xmake f --metax-gpu=y` 启用 Metax 支持。依赖 MXC 编译器（Metax
工具链）。Kernel 实现在 `src/ops/*/metax/` 目录下，使用 `.mc`（实现）和 `.mch`（头文件）扩展名。设备抽象层实现在
`src/device/metax/`。

---

## 三、构建与复现

### 3.1 环境要求

| 依赖               | 版本/说明                                                 |
|:-----------------|:------------------------------------------------------|
| Xmake            | latest（CI 使用 `xmake-io/github-action-setup-xmake@v1`） |
| C++ 编译器          | MSVC 2022 (Windows) / GCC 或 Clang (Linux)，需支持 C++17   |
| Python           | >= 3.9，需 PyTorch、Transformers、HuggingFace Hub         |
| CUDA Toolkit（可选） | 仅 NVIDIA 后端需要                                         |
| MXC 编译器（可选）      | 仅 Metax 后端需要                                          |

### 3.2 复现步骤

#### 1. 克隆仓库

```bash
git clone <repo-url>
cd llaisys-26s
```

#### 2. CPU 构建与测试

```bash
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
```

#### 3. CUDA 构建与测试

```bash
xmake f --nv-gpu=y -c
xmake
xmake install
pip install ./python/

# 运行时测试
python test/test_runtime.py --device nvidia

# 算子测试
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia

# 端到端推理
python test/test_infer.py --model <path/to/model> --test --device nvidia
```

#### 4. Metax 构建与测试

```bash
xmake f --metax-gpu=y -c
xmake
xmake install
pip install ./python/

# 运行时测试
python test/test_runtime.py --device metax

# 算子测试
python test/ops/add.py --device metax
python test/ops/argmax.py --device metax
python test/ops/embedding.py --device metax
python test/ops/linear.py --device metax
python test/ops/rms_norm.py --device metax
python test/ops/rope.py --device metax
python test/ops/self_attention.py --device metax
python test/ops/swiglu.py --device metax

# 端到端推理
python test/test_infer.py --model <path/to/model> --test --device metax
```

### 3.3 复现结果

所有测试在以下环境验证通过：

#### Linux (Ubuntu, GCC)

```text
test_runtime.py --device cpu         PASS
test_tensor.py                       PASS
test/ops/*.py (8 个算子, 3 种精度)     PASS
test_infer.py --test                 PASS  128 token 与 HuggingFace 输出完全一致
```

#### Windows (MSVC 2022)

```text
（CI 验证，同上全绿）
```

#### NVIDIA RTX（RTX 4090 D - 24 GB × 1）

```text
test_runtime.py --device nvidia      PASS
test/ops/*.py --device nvidia        PASS  全部通过
test_infer.py --test --device nvidia PASS  128 token 完全一致
```

#### Metax 曦云 C500（×1, 16 GB）

```text
test_runtime.py --device metax        PASS
test/ops/*.py --device metax          PASS  8 个算子全部通过
test_infer.py --test --device metax   PASS  128 token 完全一致
```

---

## 四、性能数据

### 4.1 端到端推理（`test/test_infer.py --test`）

DeepSeek-R1-Distill-Qwen-1.5B，prompt="Who are you?"，生成 ~85 token，确定性采样（temp=1.0, top_k=1）。token 序列与
HuggingFace Transformers 完全一致，正确性已验证。

#### CPU vs NVIDIA（RTX 4090 D - 24 GB × 1 机器）

| 设备      |  总耗时   | ms/token | 输出正确 |
|:--------|:------:|:--------:|:----:|
| CPU (同机) | 375.65s |  ~4,368  | PASS |
| NVIDIA   | 0.46s  |   ~5.3   | PASS |

> 端到端加速比约 **817×**。

#### CPU vs Metax（曦云 C500 ×1, 16 GB 机器）

| 设备      |  总耗时   | ms/token | 输出正确 |
|:--------|:------:|:--------:|:----:|
| CPU (同机) | 406.70s |  ~4,730  | PASS |
| Metax   | 1.73s  |  ~20.1   | PASS |

> 端到端加速比约 **235×**。

### 4.2 单算子 Micro-benchmark（Qwen2 1.5B decode 形状，f32）

#### CPU vs NVIDIA（RTX 4090 D - 24 GB × 1 机器）

| 算子               |  CPU (ms)  | NVIDIA (ms) |   加速比    |
|:-----------------|:----------:|:-----------:|:--------:|
| add              |   0.0012   |   0.0038    |   0.3×   |
| argmax           |   0.0940   |   0.0168    |   5.6×   |
| embedding        |   0.0011   |   0.0036    |   0.3×   |
| linear (proj)    |   2.7885   |   0.0094    |   297×   |
| linear (head)    |  280.1699  |   1.0356    |   271×   |
| linear (gate/up) |  15.0177   |   0.0239    |   628×   |
| linear (down)    |  15.3348   |   0.0188    |   816×   |
| rms_norm         |   0.0033   |   0.0045    |   0.7×   |
| rope (q)         |   0.0107   |   0.0040    |   2.7×   |
| rope (k)         |   0.0028   |   0.0040    |   0.7×   |
| self_attention   |   0.3481   |   0.0440    |   7.9×   |
| swiglu           |   0.0312   |   0.0037    |   8.4×   |
| **合计**           | **313.80** |  **1.17**   | **268×** |

> NVIDIA 的 CUDA kernel 经过 cuBLAS 优化，在矩阵运算密集的 linear 算子上优势明显。

#### CPU vs Metax（曦云 C500 ×1, 16 GB 机器）

| 算子               |  CPU (ms)  | Metax (ms) |    加速比    |
|:-----------------|:----------:|:----------:|:---------:|
| add              |   0.0014   |   0.0113   |   0.1×    |
| argmax           |   0.1105   |   0.0864   |   1.3×    |
| embedding        |   0.0011   |   0.0082   |   0.1×    |
| linear (proj)    |   1.9575   |   0.2708   |   7.2×    |
| linear (head)    |  178.2263  |   5.3792   |   33.1×   |
| linear (gate/up) |   8.9686   |   0.4760   |   18.8×   |
| linear (down)    |   9.2692   |   1.3861   |   6.7×    |
| rms_norm         |   0.0040   |   0.0137   |   0.3×    |
| rope (q)         |   0.0115   |   0.0160   |   0.7×    |
| rope (k)         |   0.0029   |   0.0157   |   0.2×    |
| self_attention   |   0.3970   |   0.1719   |   2.3×    |
| swiglu           |   0.0585   |   0.0107   |   5.5×    |
| **合计**           | **199.01** |  **7.85**  | **25.4×** |

> Metax 平台上 linear（head）是主要瓶颈，占单 token 计算时间的 ~68.6%。小算子（add、embedding、rms_norm、rope）由于 kernel launch
> 开销，在 Metax 上反而比 CPU 慢。
