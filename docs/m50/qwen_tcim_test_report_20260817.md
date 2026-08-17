# FlashRT 原生 TCIM 后端 M50/Qwen3-0.6B 测试报告

- 测试日期：2026-08-17
- 测试对象：`flash_rt.load_model(..., framework="tcim", config="qwen")`
- 测试设备：后摩 M50/XH2，device 0

## 1. 测试目的与结论

本报告验证 FlashRT 原生 TCIM Qwen 路径在 M50 上的推理正确性、会话状态、
资源释放、运行时依赖和性能。

Qwen3-0.6B 已通过全部测试。相同输入的三次 token 序列一致，连续请求未发生
状态串扰，物理 KV Cache 复位结果正确，模型关闭后没有遗留 GGUF 文件描述符。
进程仅加载 TCIM Runtime，未加载 llama.cpp 或 HLIELLama。

## 2. 被测执行链路

```text
Qwen GGUF
  -> FlashRT：GGUF 解析、tokenizer、embedding、KV Cache、prefill/decode、采样
  -> tcim_lite
  -> TCIM Runtime：执行 prefill.hmm 和 decoder.hmm
  -> M50/XH2
```

GGUF 是本次模型交付容器。FlashRT 根据文件偏移加载其中的 HMM，不调用
`libllama.so`。HMM 的离线量化与编译不属于本次运行时测试范围。

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

## 4. 测试方法

正确性测试使用 greedy sampling。短回答测试连续执行 3 次；会话隔离测试在同一
模型实例中依次请求输出 2、3、2；物理复位测试启用
`zero_kv_on_reset=True`。动态库通过 `/proc/self/maps` 检查，资源释放通过
模型文件描述符数量检查。

性能测试使用 batch 1、greedy sampling、固定输出 32 token，并在同一实例中
连续执行 3 次。32 个输出 token 对应 31 次 decoder HMM 执行，首个 token 来自
prefill logits。报告采用三次测试的中位数。

## 5. 正确性测试结果

| 测试项 | 判定标准 | 结果 |
|---|---|---|
| Qwen 专项单元测试 | GGUF 解析、API 分发、参数校验、EOG 和资源释放 | 10/10 通过 |
| GGUF 资产检查 | 必需资产存在且偏移、大小未越过文件范围 | 通过 |
| M50 模型加载 | prefill、decoder 和共享 KV Cache 正常初始化 | 通过 |
| 短回答重复性 | 相同提示连续执行 3 次 | Token IDs 3/3 一致 |
| 会话状态隔离 | 同一实例依次请求输出 2、3、2 | 分别输出 2、3、2 |
| KV Cache 物理复位 | `zero_kv_on_reset=True` 连续执行两次 | 两次输出一致 |
| EOG 状态 | 生成结束后再次调用 `decode()` | 完成状态保护生效 |
| 资源释放 | 模型加载前、加载中、`close()` 后的 GGUF 文件描述符 | 0 → 1 → 0 |
| 动态库依赖 | 检查 `/proc/self/maps` | TCIM 已加载，`libllama.so` 未加载 |
| 测试后设备状态 | 再次执行 `hm_smi` | device 0 正常 |

短回答实际结果：

```text
输入：请只输出数字2，不要输出其他内容。 /no_think
输出：<think>\n\n</think>\n\n2
Token IDs：[151667, 271, 151668, 271, 17, 151645]
```

## 6. 性能测试结果

| 指标 | 当前结果 | 说明 |
|---|---:|---|
| 模型加载时间 | 14.399 s | GGUF 解析、TCIM 图加载和 Cache 分配 |
| Prefill 端到端中位时延 | 40.580 ms | Tokenizer、embedding、输入绑定和 HMM 执行 |
| Prefill HMM 中位时延 | 26.267 ms | TCIM prefill 图执行与同步 |
| TCIM decoder 图中位吞吐 | 71.658 step/s | 31 次 decoder HMM 执行 |
| FlashRT 端到端输出吞吐 | 41.427 token/s | 包括输入绑定、logits 获取和采样 |
| Decode 图外 host 时间 | 339.823 ms / 32 token | `decode_wall_ms - decode_hmm_ms` |
| Token 序列重复性 | 3/3 一致 | `temp=0.0` |

三次端到端吞吐分别为 42.197、39.676 和 41.427 token/s。原始数据：

- [`results/qwen3_0.6b_tcim_20260817.json`](results/qwen3_0.6b_tcim_20260817.json)
- [`results/qwen3_0.6b_tcim_throughput_20260817.json`](results/qwen3_0.6b_tcim_throughput_20260817.json)

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

结果应满足：

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

调试过程和修复记录见
[`qwen_tcim_debug_report_20260817.md`](qwen_tcim_debug_report_20260817.md)。
