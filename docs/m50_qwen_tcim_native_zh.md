# FlashRT 原生 TCIM 后端在 M50 上部署 Qwen

最后核对日期：2026-08-17

## 1. 结论

Qwen3-0.6B 已通过 FlashRT 原生 TCIM 后端在后摩 M50 上完成推理。当前路径不
链接、不导入、不加载 llama.cpp 或后摩 HLIELLama：

```text
FlashRT
  -> tcim_lite
  -> TCIM Runtime
  -> M50
```

实机进程映射检查结果：

```text
libllama_mapped=false
tcim_runtime_mapped=true
```

因此，这不是用 FlashRT 包装 `llama-cli`，也不是由 FlashRT 转交 HLIELLama
完成生成。FlashRT 直接管理 tokenizer、embedding、prefill、KV Cache、decode、
采样和文本解码；TCIM Runtime 仅负责执行已经编译好的 M50 HMM 图。

## 2. 测试环境

| 项目 | 版本或配置 |
|---|---|
| 主机架构 | aarch64 |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 内核 | 5.10.226 |
| Python | 3.12.13 |
| FlashRT 分支 | `codex/m50-qwen-tcim-native` |
| M50 | HoumoNPU LQ50-24GB，24448 MiB |
| HMSW / Driver / Firmware | V1.4.0 / V1.4.0 / V1.4.0 |
| TCIM Runtime | V1.4.0 |
| `tcim_lite` | `houmo_tcim_runtime_xh2==1.4.0` |

Python 使用独立环境：

```text
/home/sky/icode/.venv-m50-runtime
```

## 3. 模型和推理产物

测试模型：

```text
/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
```

| 属性 | 值 |
|---|---|
| 模型 | Qwen3-0.6B |
| 文件大小 | 1,006,385,824 bytes |
| 目标 | XH2 / M50 |
| 配置 | batch 1，单卡 2 核 |
| HMM 版本 | V1.2.0 |
| 编译上下文长度 | 32768 |
| Prefill 长度 | 256 |

GGUF 只是当前交付件的容器。FlashRT 的轻量解析器读取容器索引，不调用
`libllama.so`。其中的主要运行资产如下：

| 资产 | GGUF 文件绝对偏移 | 大小 |
|---|---:|---:|
| FP16 embedding | 5,933,728 | 311,164,944 bytes |
| `prefill.hmm` | 317,098,688 | 656,047,800 bytes |
| `decoder.hmm` | 973,146,496 | 17,357,072 bytes |
| `tokenizer.json` | 990,503,584 | 11,422,654 bytes |

TCIM 的 `Option.set_model_offset()` 直接从原 GGUF 文件加载两个 HMM 字节区间，
不需要先复制出 600 MB 以上的模型文件。Tokenizer 配置首次加载时提取到
FlashRT 仓库的 `.cache/m50-qwen-tokenizers/`，不修改原模型。

## 4. FlashRT 实现

关键文件：

```text
flash_rt/frontends/m50/gguf_assets.py
flash_rt/frontends/m50/qwen.py
examples/m50/qwen_tcim.py
examples/m50/run_qwen_tcim.sh
```

FlashRT 原生后端执行以下工作：

1. 解析 GGUF V2/V3 元数据和 byte tensor 索引；
2. 映射 FP16 embedding，不复制完整 embedding 权重；
3. 使用 Hugging Face tokenizer 应用 Qwen chat template；
4. 通过共享 `WeightManager` 加载 prefill 和 decode HMM；
5. 将 28 层 K/V Cache 绑定到两个 HMM；
6. 执行 embedding lookup 和长度对齐；
7. 调度 prefill 和逐 token decode；
8. 在 FlashRT 内完成 greedy、top-k、top-p 和 temperature 采样；
9. 解码 token 并维护会话状态。

默认会话复位使用 `valid_length=0`。未清零区域不会进入有效注意力范围，避免了
每次物理清零约 1.75 GB KV Cache 的开销。若应用有物理清零要求，可设置
`zero_kv_on_reset=True`；该模式会增加新请求延迟。

## 5. Python API

```python
import flash_rt

model = flash_rt.load_model(
    MODEL_GGUF,
    framework="tcim",
    config="qwen",
    device_id=0,
    max_tokens=32,
    temp=0.0,
    top_k=0,
    top_p=0.0,
)

model.prefill("请只回答一个数字：一加一等于多少？", return_logits=False)
for _ in range(32):
    step = model.decode(return_text=False)
    if step["is_eog"]:
        break

print(model.get_text())
model.close()
```

也可使用 `model.generate(prompt)` 完成一次生成。

## 6. 运行方法

```bash
cd /home/sky/icode/FlashRT

export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MAX_TOKENS=32
export REPEAT=3

examples/m50/run_qwen_tcim.sh
```

脚本只配置 TCIM 和后摩 HAL 动态库路径：

```text
/opt/houmo-tcim-runtime-1.4.0/lib
/usr/local/houmo-sdk/hal/lib
```

运行命令不包含 HLIELLama 目录，不包含 `libllama.so`，也不需要构建原先的
llama.cpp C++ Provider。

## 7. 实机结果

测试参数为 batch 1、prefill 256、最大生成 32 token、greedy sampling、重复
3 次。结果保存于 `benchmarks/m50_qwen_tcim_20260817.json`。

| 指标 | 结果 |
|---|---:|
| 模型加载 | 14.556 s |
| Prefill HMM 中位数 | 26.715 ms |
| Decode 端到端吞吐中位数 | 40.485 token/s |
| Token 序列重复性 | 3 次完全一致 |
| `libllama.so` | 未加载 |
| `libtcim_runtime_lite.so` | 已加载 |

短回答功能测试输入：

```text
请只输出数字2，不要输出其他内容。 /no_think
```

三次连续请求均输出：

```text
<think>

</think>

2
```

交替执行“输出 2”“输出 3”“输出 2”时结果分别为 2、3、2，说明默认逻辑复位
下未观察到跨请求 KV Cache 污染。

## 8. 验证方法

单元测试：

```bash
cd /home/sky/icode/FlashRT
/home/sky/icode/.venv-m50-runtime/bin/pytest -q \
  tests/test_m50_qwen_frontend.py \
  tests/test_m50_pi05_frontend.py \
  tests/test_action_transforms.py
```

当前结果为 11 项测试全部通过。

运行时依赖检查由 `examples/m50/qwen_tcim.py` 自动完成。它读取
`/proc/self/maps`，结果文件中必须满足：

```json
{
  "libllama_mapped": false,
  "tcim_runtime_mapped": true
}
```

## 9. 当前边界

- 当前原生后端已实机支持该 Qwen3-0.6B M50 交付件。
- 单次 prompt 不能超过编译好的 256-token prefill 长度；decode 上下文上限为
  32768。
- 当前为 batch 1、单会话实例；并发请求应创建独立实例并进行资源评估。
- Qwen3.6-35B-A3B 使用混合注意力/状态空间结构，除 K/V Cache 外还有卷积和
  SSM 状态。其原生 TCIM 状态绑定尚未实现，不能复用本后端直接运行。
- 本机仍只负责运行已经量化编译的 HMM；从 PyTorch/Hugging Face 权重生成 HMM
  仍需后摩离线量化编译工具链。
