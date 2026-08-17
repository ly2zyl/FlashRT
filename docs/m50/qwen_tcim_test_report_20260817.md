# FlashRT 原生 TCIM 后端 M50/Qwen3-0.6B 测试报告

- 测试日期：2026-08-17
- 测试对象：`flash_rt.load_model(..., framework="tcim", config="qwen")`
- 测试设备：后摩 M50/XH2，device 0

## 1. 结论

修正代码链路后，Qwen3-0.6B 已在 M50 上通过功能、状态隔离、资源释放、依赖
检查和重复性能测试。三次推理输出 token 完全一致，连续请求结果正确，进程未
加载 llama.cpp 或 HLIELLama。

本轮定位并修复了 KV Cache 绑定方向、逐 token 临时对象、采样缓冲区、EOG
状态保护、参数校验和 GGUF 内存映射释放问题。32-token 端到端吞吐由修复前的
38.123 token/s 提升到 41.427 token/s。

## 2. 被测执行链路

```text
Qwen GGUF
  -> FlashRT：GGUF 解析、tokenizer、embedding、KV Cache、prefill/decode、采样
  -> tcim_lite
  -> TCIM Runtime：执行 prefill.hmm 和 decoder.hmm
  -> M50/XH2
```

GGUF 仅作为模型交付容器。FlashRT 根据文件偏移加载其中的 HMM，不调用
`libllama.so`。HMM 的离线量化与编译不属于本次运行时链路。

## 3. 测试环境和模型

| 项目 | 实际配置 |
|---|---|
| 主机 | aarch64，Ubuntu 22.04.5 LTS，kernel 5.10.226 |
| Python | 3.12.13，`/home/sky/icode/.venv-m50-runtime` |
| M50 | HoumoNPU LQ50-24GB，device 0 |
| 后摩软件 | HMSW、Driver、Firmware、TCIM Runtime 均为 V1.4.0 |
| Python 依赖 | `houmo_tcim_runtime_xh2==1.4.0`，`transformers==4.57.6`，`numpy==2.2.6` |
| 模型 | Qwen3-0.6B，GGUF V3，HMM V1.2.0，batch 1，单卡 2 核 |
| 模型文件 | `HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf` |
| 输入规格 | prefill 256 token，context 32768 token，embedding 151936 × 1024 FP16 |

模型文件大小为 1,006,385,824 bytes。`quant_embedding.bin`、`prefill.hmm`、
`decoder.hmm` 和 tokenizer 资产均通过文件范围检查。

## 4. 调试发现与修正

| 项目 | 原因 | 修正 | 实机验证 |
|---|---|---|---|
| KV Cache 所有权方向错误 | decoder 的 Cache 输入被设为 dummy tensor，原实现却从 decoder 取 Cache 再绑定给 prefill，不符合 TCIM 接口契约 | 改为由 prefill 分配 Cache，再通过 `decode.set_input()` 绑定给 decoder | 连续 2→3→2 请求正确，无串话 |
| Decode 热路径重复分配 | 每 token 新建 embedding、valid length 和 current length 数组 | 初始化时预分配并循环复用 | 输出不变，host 开销下降 |
| Greedy 采样临时分配 | 每 token 创建新的 FP32 logits 数组 | 使用固定 FP32 采样缓冲区；缓存输入输出名称 | 32-token 吞吐提升 8.7% |
| EOG 后仍可继续 decode | 只记录 token 数，没有完成状态 | 增加 `_finished` 状态，EOG 或 token 预算结束后拒绝再次 decode | 专项单元测试通过 |
| 运行参数约束不完整 | 负 device、temperature、top-k 或越界 top-p 未提前拒绝 | 构造阶段统一校验 | 5 组非法参数单元测试通过 |
| GGUF mmap 未显式关闭 | `close()` 只释放 TCIM module，没有关闭 embedding memmap | 关闭底层 mmap 并清空引用 | 模型文件描述符由加载时 1 个降为关闭后 0 个 |

TCIM 启动时打印的 backend warning 也进行了核查。后摩官方 YOLO、算子和
`tcim_perf` 工具使用环境变量选择 `Xh2HalBackend` 时会打印相同信息，当前
`tcim_lite.runtime.Option` 也没有 backend setter。因此该信息属于 TCIM Runtime
的统一回退提示，不是 FlashRT 代码错误。

## 5. 功能测试结果

| 测试项 | 判定标准 | 结果 |
|---|---|---|
| Qwen 专项单元测试 | GGUF 解析、API 分发、参数校验、EOG 和资源释放 | 10/10 通过 |
| M50 加载 | 两个 HMM 和共享 KV Cache 正常初始化 | 通过 |
| 短回答重复性 | 相同提示连续执行 3 次 | 输出 token 3/3 一致 |
| 会话状态隔离 | 同一实例依次请求输出 2、3、2 | 分别输出 2、3、2 |
| KV Cache 物理复位 | `zero_kv_on_reset=True` 连续执行两次 | 两次输出一致 |
| EOG 状态 | 生成结束后再次调用 `decode()` | 被完成状态保护拦截 |
| 资源释放 | 比较模型加载前、加载中和 `close()` 后的文件描述符 | 0 → 1 → 0 |
| 动态库依赖 | 检查 `/proc/self/maps` | TCIM 已加载，`libllama.so` 未加载 |
| 测试后设备状态 | 再次执行 `hm_smi` | device 0 正常 |

短回答实际输出：

```text
输入：请只输出数字2，不要输出其他内容。 /no_think
输出：<think>\n\n</think>\n\n2
Token IDs：[151667, 271, 151668, 271, 17, 151645]
```

## 6. 性能测试结果

测试参数为 batch 1、greedy sampling、固定输出 32 token、连续执行 3 次。
32 个输出 token 对应 31 次 decoder HMM 执行；首个 token 来自 prefill logits。

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 模型加载时间 | 14.388 s | 14.399 s | 基本不变 |
| Prefill HMM 中位时延 | 26.154 ms | 26.267 ms | 基本不变 |
| TCIM decoder 图中位吞吐 | 70.855 step/s | 71.658 step/s | +1.1% |
| FlashRT 端到端输出吞吐 | 38.123 token/s | 41.427 token/s | +8.7% |
| 图外 host 时间 | 401.871 ms | 339.823 ms | -15.4% |
| Token 序列重复性 | 3/3 一致 | 3/3 一致 | 不变 |

修复后三次端到端吞吐分别为 42.197、39.676 和 41.427 token/s。原始结果见：

- [`results/qwen3_0.6b_tcim_20260817.json`](results/qwen3_0.6b_tcim_20260817.json)
- [`results/qwen3_0.6b_tcim_throughput_20260817.json`](results/qwen3_0.6b_tcim_throughput_20260817.json)

当前 HMM 每步输出形状为 `[1, 1, 151936]` 的 FP16 完整词表 logits。CPU 采样
必须在每个 token 取回约 304 KiB logits，同时完成三个输入绑定，因此图外时间
不能完全消除。进一步明显提速需要新增设备端 sampler/argmax HMM，或者让 TCIM
只回传候选 token；这需要新的编译产物，不是现有 Python 调度代码可以独立完成。

## 7. 复现命令

```bash
cd /home/sky/icode/FlashRT

PYTHONDONTWRITEBYTECODE=1 \
/home/sky/icode/.venv-m50-runtime/bin/pytest -q \
  tests/test_m50_qwen_frontend.py

export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MAX_TOKENS=32
export REPEAT=3
export PROMPT='请用中文详细说明矩阵乘法的计算过程，并给出一个例子。 /no_think'
export RESULT_JSON=/home/sky/icode/FlashRT/docs/m50/results/qwen3_0.6b_tcim_throughput_20260817.json

examples/m50/run_qwen_tcim.sh
```

结果应同时满足：

```json
{
  "libllama_mapped": false,
  "tcim_runtime_mapped": true,
  "token_ids_repeatable": true
}
```

## 8. 适用边界

- 当前验证对象仅为该 Qwen3-0.6B M50 交付件；
- prompt 上限为 256 token，decode context 上限为 32768 token；
- 当前实现为 batch 1、单会话实例；
- Qwen3.6-35B-A3B 需要额外绑定卷积和 SSM 状态，不能直接复用本实现；
- aarch64 M50 机器只运行已有 HMM，不能替代 x86_64 离线量化编译工具链。
