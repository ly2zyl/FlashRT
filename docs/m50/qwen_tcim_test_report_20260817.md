# FlashRT 原生 TCIM 后端 M50/Qwen3-0.6B 测试报告

- 测试日期：2026-08-17
- 测试对象：FlashRT `tcim/qwen` 前端
- 目标设备：后摩 M50/XH2，device 0

## 1. 测试目的

验证 FlashRT 在不使用 llama.cpp 和 HLIELLama 的条件下，能否直接管理 Qwen
推理流程，并通过 `tcim_lite` 调度既有 HMM 在 M50 上稳定执行。测试同时检查
输入边界、跨请求状态、重复性、运行时依赖和端到端性能。

## 2. 测试结论

本次测试未发现阻断部署的功能问题。Qwen3-0.6B 已完成真实 M50 推理，输出
正确且可重复，连续请求未观察到 KV Cache 串话，进程未加载 `libllama.so`。

当前主要限制是性能而非正确性。32-token 测试中，TCIM decode 图执行中位吞吐
为 70.855 step/s，FlashRT 端到端输出吞吐为 38.123 token/s。逐 token 的输入
绑定、输出获取、全词表采样和 Python 调度合计占用 401.871 ms，后续优化应优先
减少 host/device 同步与 Python 热路径开销。

## 3. 被测链路与责任边界

```text
Qwen GGUF
  -> FlashRT GGUF 索引解析
  -> tokenizer 与 embedding lookup
  -> FlashRT prefill/decode/KV Cache 调度与采样
  -> tcim_lite
  -> TCIM Runtime 执行 prefill.hmm、decoder.hmm
  -> M50
```

GGUF 是本次后摩交付件的容器，并不表示由 llama.cpp 执行。FlashRT 使用文件
偏移直接将其中的 HMM 交给 TCIM。量化、图编译和 HMM 生成属于离线工具链，
不在本次运行时测试范围内。

## 4. 测试环境

| 项目 | 配置 |
|---|---|
| 主机 | aarch64，Ubuntu 22.04.5 LTS，kernel 5.10.226 |
| Python | 3.12.13，独立环境 `/home/sky/icode/.venv-m50-runtime` |
| M50 | HoumoNPU LQ50-24GB，device 0，单卡可见 |
| 后摩软件 | HMSW V1.4.0，Driver V1.4.0，Firmware V1.4.0 |
| TCIM | Runtime V1.4.0，`houmo_tcim_runtime_xh2==1.4.0` |
| Python 依赖 | `transformers==4.57.6`，`numpy==2.2.6`，`pytest==9.1.1` |
| FlashRT | 分支 `codex/m50-qwen-tcim-native`，以报告提交的 Git HEAD 为准 |

测试模型：

```text
/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
```

| 模型属性 | 实际值 |
|---|---|
| 名称与目标 | Qwen3-0.6B，XH2，batch 1，单卡 2 核 |
| 文件大小 | 1,006,385,824 bytes |
| GGUF / HMM | GGUF V3，HMM V1.2.0，`is_hmm=true` |
| Prefill / Context | 256 / 32768 token |
| Embedding | FP16，151936 × 1024 |

GGUF 资产边界检查结果：

| 资产 | 文件偏移 | 大小 | 边界检查 |
|---|---:|---:|---|
| `quant_embedding.bin` | 5,933,728 | 311,164,944 bytes | 通过 |
| `prefill.hmm` | 317,098,688 | 656,047,800 bytes | 通过 |
| `decoder.hmm` | 973,146,496 | 17,357,072 bytes | 通过 |
| `tokenizer.json` | 990,503,584 | 11,422,654 bytes | 通过 |

## 5. 测试项目与结果

| 编号 | 测试项目 | 方法与判定条件 | 结果 |
|---:|---|---|---|
| 1 | M50 与运行时可见性 | `hm_smi` 正常；`get_device_num("Xh2HalBackend") == 1` | 通过 |
| 2 | GGUF 解析与边界 | 解析 V3 元数据；4 个必需资产均在文件范围内 | 通过 |
| 3 | API 分发 | `framework="tcim", config="qwen"` 创建 M50 前端；错误配置被拒绝 | 通过 |
| 4 | 启动脚本校验 | 指定不存在的 GGUF | 返回码 2，错误信息明确 |
| 5 | 单元与回归 | M50 Qwen、Pi0.5 资源分发、动作变换共 11 项 | 11/11 通过 |
| 6 | CMake 配置回归 | 关闭 CUDA 与执行器后完成 configure | 通过；仅提示未构建可选 pybind11 模块 |
| 7 | 文本输入功能 | 指令仅输出数字 2，连续 3 次 token 序列一致 | 通过 |
| 8 | Token 输入功能 | 同一 prompt 改用 token IDs 输入 | 输出与文本输入一致 |
| 9 | 会话隔离 | 同一实例依次请求 2、3、2 | 分别输出 2、3、2，无可见串话 |
| 10 | 物理 KV 清零 | `zero_kv_on_reset=True` 连续请求两次 | 两次结果一致 |
| 11 | 输入与生命周期边界 | 257-token prompt；关闭后再次 `reset()` | 均按预期抛出异常 |
| 12 | 运行时依赖 | 检查 `/proc/self/maps` | TCIM 已加载，`libllama.so` 未加载 |
| 13 | 设备测试后状态 | 再次执行 `hm_smi` | device 0 正常，无驱动或固件错误 |

功能测试实际输出：

```text
输入：请只输出数字2，不要输出其他内容。 /no_think
输出：<think>\n\n</think>\n\n2
Token IDs：[151667, 271, 151668, 271, 17, 151645]
```

交替请求输出为：

```text
2 -> 3 -> 2
```

边界检查错误信息为：

```text
ValueError: prompt requires 1..256 tokens, got 257
RuntimeError: QwenM50Frontend is closed
```

## 6. 性能测试结果

性能测试使用 greedy sampling、batch 1、最大输出 32 token，每项连续执行 3
次。短回答用于功能验证；矩阵乘法说明提示词在 32 token 时未遇到 EOG，用于
稳定吞吐统计。原始数据位于 [`results/`](results/)。

| 指标 | 短回答测试 | 32-token 吞吐测试 | 说明 |
|---|---:|---:|---|
| 模型加载时间 | 14.342 s | 14.388 s | 包含 GGUF 解析、TCIM 图加载和缓存分配 |
| Prefill HMM 中位时延 | 26.238 ms | 26.154 ms | 仅 TCIM prefill 图执行与同步 |
| Prefill 端到端中位时延 | 38.880 ms | 40.057 ms | 包含 tokenizer、embedding 与输入绑定；首轮冷启动约 110 ms |
| TCIM decode 图中位吞吐 | 70.690 step/s | 70.855 step/s | 仅实际执行的 decode HMM；32 个输出对应 31 次图执行 |
| FlashRT 端到端输出吞吐 | 44.386 token/s | 38.123 token/s | 包含采样、张量获取和 Python 调度 |
| Decode host 调度中位开销 | 64.245 ms / 6 token | 401.871 ms / 32 token | `decode_wall_ms - decode_hmm_ms` |
| Token 序列重复性 | 3/3 一致 | 3/3 一致 | `temp=0.0` |

32-token 三次端到端输出吞吐分别为 35.365、41.715 和 38.123 token/s；TCIM
图执行吞吐分别为 70.165、72.237 和 70.855 step/s。TCIM 图本身波动较小，
端到端波动主要出现在图执行之外的 host 路径。

## 7. 问题与风险分析

### 7.1 未发现的阻断问题

- 未发现输出错误、HMM 加载失败、设备掉卡或运行时崩溃；
- 未发现默认逻辑复位导致的跨请求 KV Cache 污染；
- 未发现 HLIELLama 或 `libllama.so` 被间接加载；
- 文本输入和 token 输入得到一致结果。

### 7.2 已确认的性能限制

端到端 decode 明显慢于 TCIM 图执行。当前每步由 Python 完成输入绑定、TCIM
调用、输出获取、全词表采样和状态更新，这些操作在 32-token 测试中合计约占
decode wall time 的 48%。这不影响正确性，但说明当前实现尚未充分发挥 M50 图
执行吞吐。

后续优化建议按以下顺序验证：

1. 将 greedy/top-k/top-p 采样下沉到设备或 TCIM 可执行图；
2. 避免每 token 获取完整 logits，优先只回传候选 token；
3. 合并输入绑定与同步操作，减少 Python/C++ 边界调用；
4. 增加长序列和多并发基准，确认 host 开销比例是否随负载变化。

### 7.3 使用限制

- 默认逻辑复位通过 `valid_length=0` 隔离历史 KV，不物理清零约 1.75 GiB Cache；
- 物理清零模式功能正常，但第二次短请求端到端耗时约 777 ms，不能作为默认
  低时延配置；
- 当前 GGUF 的 prompt 上限为 256 token；
- Qwen3.6-35B-A3B 的卷积与 SSM 状态绑定尚未实现；
- HMM 必须由匹配的离线量化编译工具链生成，本机运行时不能替代该工具链。

### 7.4 非故障提示

- TCIM 启动时会提示使用环境变量中的 `Xh2HalBackend`，随后提示空 backend
  名称回退到该默认值；本次所有测试均正常完成；
- 系统 `PATH` 中没有 `cmake`，应使用虚拟环境内的
  `/home/sky/icode/.venv-m50-runtime/bin/cmake`；
- CMake 提示基础 Python 3.13 缺少 `pybind11` 时会只配置 C ABI。本 Qwen 路径
  是 Python 直接调用 `tcim_lite`，不依赖该可选模块。

## 8. 复现命令

```bash
cd /home/sky/icode/FlashRT

/home/sky/icode/.venv-m50-runtime/bin/pytest -q \
  tests/test_m50_qwen_frontend.py \
  tests/test_m50_pi05_frontend.py \
  tests/test_action_transforms.py

export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MAX_TOKENS=32
export REPEAT=3
export PROMPT='请用中文详细说明矩阵乘法的计算过程，并给出一个例子。 /no_think'
export RESULT_JSON=/home/sky/icode/FlashRT/docs/m50/results/qwen3_0.6b_tcim_throughput_20260817.json

examples/m50/run_qwen_tcim.sh
```

运行结果必须同时满足：

```json
{
  "libllama_mapped": false,
  "tcim_runtime_mapped": true,
  "token_ids_repeatable": true
}
```

## 9. 最终判定

FlashRT 原生 TCIM Qwen3-0.6B 路径满足当前 M50 单会话功能部署要求，可作为
后续优化基线。当前结果证明的是“FlashRT 管理推理流程并调用 TCIM 执行已有
HMM”，不是“FlashRT 已实现 M50 离线量化编译器”。在扩大模型范围或用于高
吞吐服务前，应先处理逐 token host 调度开销，并补充长上下文与并发压力测试。
