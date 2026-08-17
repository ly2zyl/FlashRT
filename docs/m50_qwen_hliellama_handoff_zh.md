# FlashRT 在后摩 M50 上部署 Qwen

最后核对日期：2026-08-17

## 1. 部署结论

FlashRT 已通过原生 C ABI Provider 在后摩 M50 上完成 Qwen 文本模型部署，已验证：

- Qwen3-0.6B：整句推理和 `prefill/decode` 分阶段推理；
- Qwen3.6-35B-A3B：`prefill/decode` 分阶段推理；
- 同一进程连续请求时的 KV Cache 隔离；
- 分阶段输出与整句输出一致，重复测试的 token 序列一致；
- 跳过 Python 侧全量 logits 拷贝和逐 token 累计文本拷贝；
- 使用预校验 SHA-256 缩短大模型启动时间。

该实现不启动 `llama-cli` 子进程。应用通过 `flash_rt.load_model()` 进入 FlashRT
稳定模型运行时 ABI，再调用后摩运行库和 M50。

## 2. 使用约束

- FlashRT 源码、构建目录、测试结果和虚拟环境位于 `/home/sky/icode`。
- `/home/sky/houmo-HLIELLama-xh2` 和现有 GGUF 模型按只读资源使用。
- 不修改系统 Python、Conda 基础环境、后摩驱动或他人模型文件。
- Python 环境为 `/home/sky/icode/.venv-m50-runtime`。

## 3. 软件和硬件环境

| 项目 | 版本或配置 |
|---|---|
| 主机架构 | aarch64 |
| 操作系统 | Ubuntu 22.04.5 LTS（Jammy） |
| 内核 | 5.10.226 |
| Python | 3.12.13 |
| CMake | 3.31.10 |
| M50 设备 | HoumoNPU LQ50-24GB，24448 MiB |
| HMSW / 驱动 / 固件 | V1.4.0 / V1.4.0 / V1.4.0 |
| HLIELLama | llama.cpp 2.1.1，lanyue 2.1.0，TCIM 1.3.0 |
| FlashRT 分支 | `codex/m50-qwen-hliellama-handoff` |

本机 `/etc/os-release` 返回 `VERSION_ID="22.04"`，因此报告采用 Ubuntu 22.04.5，
不是“24.02”。

## 4. 推理链路和职责边界

```text
FlashRT Python API
  -> Houmo LLM Frontend
  -> frt_model_runtime_v1
  -> FlashRT Houmo Llama C++ Engine
  -> 后摩 libllama.so
  -> GGUF 内嵌 prefill.hmm / decoder.hmm
  -> TCIM Runtime
  -> M50
```

各层职责如下：

| 层级 | 主要职责 |
|---|---|
| FlashRT | 模型生命周期、输入输出接口、显式 reset/prefill/decode 调度、生成长度控制、会话隔离、性能统计 |
| FlashRT Houmo Engine | 调用后摩 Llama C API，保持模型常驻，组织 prompt/token 输入和逐 token 解码 |
| 后摩 HLIELLama / `libllama.so` | 解析 GGUF、应用 chat template、tokenizer 与采样基础能力、对接后摩 HMM 后端 |
| TCIM Runtime | 设备内存和执行资源管理，在 M50 上执行已编译的 HMM 图 |
| 离线工具链 | 将原始模型量化、编译为 HMM；不参与本机运行时推理 |

当前 FlashRT 已接管推理会话和阶段调度，但没有重新实现后摩 `libllama.so` 中的
GGUF、tokenizer、HMM 后端和 TCIM 接口。因此准确表述是“FlashRT 调度和优化、
后摩运行库执行 M50 后端”，不能表述为 FlashRT 已完全替代 HLIELLama。

## 5. 测试模型

| 模型 | 文件大小 | 量化/编译信息 | SHA-256 |
|---|---:|---|---|
| Qwen3-0.6B | 1,006,385,824 B | XH2，1 卡 2 核，HMM v1.2.0 | `1fb11253f607e6209e77ae32a8b01318142afcccacd4f61c65f6145076b7dd2a` |
| Qwen3.6-35B-A3B | 21,263,621,984 B | W4A8，XH2，1 卡 2 核，HMM v1.4.0 | `43a40e4bae3de2b8b82946e4aa2da20f07d4ab793d0a3efe9db56c208c3729f8` |

模型路径：

```text
/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
/home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf
```

这里的 GGUF 是带后摩扩展内容的模型容器。除模型架构、tokenizer 和配置外，
容器内还包含已离线量化编译的 `prefill.hmm`、`decoder.hmm` 和量化 embedding。
本机只执行这些产物，不在运行时重新量化或编译。

## 6. FlashRT 实现

关键文件：

```text
cpp/providers/llama_cpp/src/houmo_llama_engine.cpp
cpp/providers/llama_cpp/src/llm_runtime.cpp
flash_rt/frontends/houmo_llama/llm.py
flash_rt/frontends/jetson_pi/llm.py
examples/m50/qwen_hliellama.py
examples/m50/run_qwen_hliellama.sh
```

Provider 支持两种调用方式：

- `generate(prompt)`：一次完成 prefill 和 decode；
- `prefill()` + `decode()`：由 FlashRT 显式调度生成过程。

分阶段路径支持 prompt 字符串或 `int32` token 数组。性能测试默认
`prefill(return_logits=False)`，避免将完整词表 logits 拷贝到 Python；逐 token
调用 `decode(return_text=False)`，结束后只拷贝一次累计文本。

后摩 HMM 后端的内部 KV 偏移不能通过公开的 `llama_memory_clear()` 完全复位。
实现会在新请求到来时重建 `llama_context`，但保留已加载的模型权重，从而保证
连续请求之间无 KV 泄漏。

## 7. 构建

```bash
cd /home/sky/icode/FlashRT

/home/sky/icode/.venv-m50-runtime/bin/cmake \
  -S cpp -B build/houmo-llama \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DFLASHRT_CPP_WITH_EXEC=OFF \
  -DFLASHRT_CPP_WITH_CUDA_STAGING=OFF \
  -DFLASHRT_CPP_WITH_CUDA_KERNELS=OFF \
  -DFLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON \
  -DFLASHRT_CPP_WITH_HOUMO_LLAMA=ON \
  -DHoumoLlama_ROOT=/home/sky/houmo-HLIELLama-xh2

/home/sky/icode/.venv-m50-runtime/bin/cmake \
  --build build/houmo-llama \
  --target flashrt_cpp_llama_cpp_provider_c \
  --parallel 2
```

构建产物：

```text
build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so
build/houmo-llama/runtime/libflashrt_runtime.so
```

## 8. 运行

推荐使用封装脚本。Qwen3-0.6B 示例：

```bash
cd /home/sky/icode/FlashRT

export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MODEL_SHA256_FILE=/home/sky/icode/FlashRT/artifacts/m50_qwen/cache/qwen3_0.6b.sha256
export MODE=both
export MAX_TOKENS=32
export REPEAT=3

examples/m50/run_qwen_hliellama.sh
```

Qwen3.6-35B-A3B 只需替换 `MODEL_GGUF` 和 `MODEL_SHA256_FILE`。脚本会设置独立
Python、`XDG_DATA_HOME` 和动态库路径。系统 `libstdc++.so.6` 必须排在后摩目录
之前；后摩目录内的旧版本缺少 Provider 所需的 `GLIBCXX_3.4.29`。

也可以直接使用 Python API：

```python
import flash_rt

model = flash_rt.load_model(
    MODEL_GGUF,
    framework="houmo_llama",
    config="llm",
    backend="houmo",
    n_ctx=512,
    n_threads=8,
    temp=0.0,
    top_k=0,
    top_p=0.0,
    max_tokens=32,
    model_identity=PREVERIFIED_SHA256,
    lib_path=PROVIDER_DSO,
)

model.prefill("请只回答一个数字：一加一等于多少？", return_logits=False)
for _ in range(32):
    step = model.decode(return_text=False)
    if step["is_eog"]:
        break
print(model.get_text())
model.close()
```

## 9. 模型标识预校验

默认情况下，Provider 会计算整份 GGUF 的 SHA-256，用于 FlashRT 部署标识。
21 GB 模型每次启动都计算哈希会产生明显开销。可在首次部署时生成并校验：

```bash
sha256sum MODEL.gguf > MODEL.gguf.sha256
sha256sum -c MODEL.gguf.sha256
```

校验通过且模型保持只读、内容未变时，可将摘要文件通过
`--model-sha256-file` 或 `model_identity=` 传入。该参数是可信输入，不会再次读取
整份文件验证；模型被替换或修改后必须重新计算。

## 10. 测试结果

测试日期为 2026-08-17，单张 M50、上下文长度 512、贪心采样。结构化摘要保存于
`benchmarks/m50_qwen_20260817.json`。

| 模型与场景 | 加载时间 | Prefill | Decode | 功能结果 |
|---|---:|---:|---:|---|
| Qwen3-0.6B，32 token，3 次中位数 | 12.274 s | 38.431 ms | 37.448 token/s | token 序列重复；one-shot 与 staged 文本一致 |
| Qwen3-0.6B，预校验标识，功能测试 | 6.804 s | 36.674 ms | 6 token 完整结束 | 输出数字 2 |
| Qwen3.6-35B-A3B，功能测试 | 299.255 s | 276.346 ms | 20.713 token/s | 6 token 完整结束；输出数字 2 |
| Qwen3.6-35B-A3B，预校验启动测试 | 124.044 s | 281.563 ms | 8-token 短测 16.666 token/s | FlashRT staged 路径正常 |

Qwen3-0.6B 的后摩 `llama-cli` 参考生成速度为 38.8 token/s，FlashRT 32-token
测试中位数为 37.448 token/s，相差约 3.5%。35B 预校验启动测试只生成 8 个
token，固定开销占比较高，不用于替代完整结束测试的 20.713 token/s 结论。

加载时间受文件页缓存和系统负载影响。预校验结果用于说明重复全文件哈希已被
消除，不应作为设备算力指标。

## 11. 验证命令

```bash
cd /home/sky/icode/FlashRT

XDG_DATA_HOME=/home/sky/icode/.local-share hm_smi

/home/sky/icode/.venv-m50-runtime/bin/pytest -q \
  tests/test_m50_qwen_frontend.py \
  tests/test_m50_pi05_frontend.py \
  tests/test_action_transforms.py
```

当前结果：10 项测试全部通过。Qwen3-0.6B 和 Qwen3.6-35B-A3B 均已在 M50
实机完成 FlashRT 端到端推理。

## 12. 已知边界

- 当前模型已经包含 M50 HMM；本机不需要量化器和编译器。
- 若从 Hugging Face/PyTorch/ONNX 权重开始，需要在受支持的 x86 转换环境中
  完成后摩量化编译，再将生成的 HMM/GGUF 交付到本机。
- 当前实现面向 batch 1、单会话。并发请求需要独立 Provider 实例，尚未进行
  多会话吞吐测试。
- 若目标是完全移除后摩 `libllama.so`，还需在 FlashRT 内实现 GGUF 扩展解析、
  tokenizer/chat template、采样、KV 状态和 TCIM/HMM 调用接口；不属于本次已完成范围。
