# FlashRT TCIM 后端 M50/Qwen3-0.6B 测试报告

- 测试日期：2026-08-17
- 被测接口：`flash_rt.load_model(..., framework="tcim", config="qwen")`
- 测试设备：后摩 M50/XH2，device 0

## 1. 硬件环境

测试主机为 ARM64 嵌入式计算平台，安装一张后摩 LQ50-24GB 加速卡。测试前后
使用 `hm_smi -a -vvv` 检查设备状态，M50 均工作在 performance 模式，IPU 与
Core 频率均为 1300 MHz。

| 项目 | 测试配置 |
|---|---|
| 主机架构 | aarch64，8 核 ARM Cortex-A55/Cortex-A76 |
| 主机内存 | 15.6 GiB，无 Swap |
| 操作系统 | Ubuntu 22.04.5 LTS |
| Linux 内核 | 5.10.226 |
| 加速卡 | Houmo LQ50-24GB，device 0，2 个计算核 |
| M50 板载内存 | 24,448 MB |
| M50 工作频率 | IPU 1300 MHz，Core 1300 MHz |
| 频率策略 | `performance`，PLL 锁定 1300 MHz |
| 可用计算核 | `0x11`，Core 0 和 Core 1 均可用 |
| 主机接口 | 8.0 GT/s × 2 lane |
| 后摩软件版本 | HMSW V1.4.0，Driver V1.4.0，Firmware V1.4.0 |

频率是本报告性能数据的测试条件。不同 DVFS 模式、IPU 频率或可用核数量下的
数据不能直接与本报告比较。

## 2. 环境搭建

### 2.1 虚拟环境创建

全部 Python 安装和 FlashRT 调试均在 `/home/sky/icode` 下的独立虚拟环境中
完成。该环境实际使用以下命令创建：

```bash
cd /home/sky/icode
/home/sky/miniforge3/envs/yolo11m-hm/bin/python \
  -m venv --system-site-packages .venv-m50-runtime
source /home/sky/icode/.venv-m50-runtime/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /home/sky/icode/FlashRT
```

虚拟环境的 Python 为 3.12.13。使用 `--system-site-packages` 是因为 M50 的
`tcim_lite` 和 HMatC 已安装在只读复用的 `yolo11m-hm` 基础环境中，普通 PyPI
不能提供等价的 aarch64 后摩运行时。后续 `pip` 写入目标为
`.venv-m50-runtime`，不会修改基础环境；但该环境会读取基础环境中的包，因此
不是完全封闭的依赖副本。

### 2.2 软件和依赖版本

| 组件 | 版本或路径 |
|---|---|
| Python | 3.12.13 |
| FlashRT | 当前分支 `codex/m50-qwen-tcim-native`，editable install |
| `houmo_tcim_runtime_xh2` | 1.4.0 |
| TCIM Runtime | V1.4.0，`/opt/houmo-tcim-runtime-1.4.0` |
| HMatC | 1.4.0.dev0 |
| NumPy | 2.2.6 |
| Transformers | 4.57.6 |
| PyTorch / TorchVision | 2.13.0 / 0.28.0 |
| Pytest | 9.1.1 |
| HAL 动态库 | `/usr/local/houmo-sdk/hal/lib` |

执行 M50 测试前设置运行时环境：

```bash
cd /home/sky/icode/FlashRT
source /home/sky/icode/.venv-m50-runtime/bin/activate
export PYTHONPATH=/home/sky/icode/FlashRT
export XDG_DATA_HOME=/home/sky/icode/FlashRT/.cache/m50/xdg
export TCIM_BACKEND=Xh2HalBackend
export LD_LIBRARY_PATH=/opt/houmo-tcim-runtime-1.4.0/lib:/usr/local/houmo-sdk/hal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## 3. 测试方法和测试对象

### 3.1 测试对象

本次测试对象为已经针对 XH2 编译的 Qwen3-0.6B 部署交付件。模型文件只读
使用，测试代码和结果均保存在 `/home/sky/icode/FlashRT` 中。

| 项目 | 模型信息 |
|---|---|
| 模型文件 | `/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf` |
| 文件大小 | 1,006,385,824 bytes |
| SHA-256 | `1fb11253f607e6209e77ae32a8b01318142afcccacd4f61c65f6145076b7dd2a` |
| 容器信息 | GGUF V3，Qwen3-0.6B，XH2 HMM V1.2.0 |
| 编译规格 | batch 1，1 张卡，2 个计算核，Prefill 256，Context 32768 |
| Embedding | 151,936 × 1,024，FP16，311,164,944 bytes |
| Prefill 计算图 | `prefill.hmm`，656,047,800 bytes |
| Decode 计算图 | `decoder.hmm`，17,357,072 bytes |
| KV Cache | 28 层 K/V，共 56 个 INT8 Cache Tensor |

执行链路如下：

```text
Qwen GGUF
  -> FlashRT：GGUF 解析、Tokenizer、Embedding、KV Cache、调度、采样和文本解码
  -> tcim_lite / TCIM Runtime：执行 Prefill 与 Decode 计算图
  -> M50/XH2
```

进程不导入或加载 `libllama.so`。模型的离线量化和编译已经包含在交付件中，不
属于本次 ARM64 运行时测试范围。

### 3.2 正确性测试方法

正确性分为无硬件单元测试和 M50 实机测试两部分。

单元测试检查 GGUF 资产索引、FlashRT API 分发、参数边界、按名称绑定 HMM
输入、KV Cache 清零开关、EOG 状态、首 token 调度和资源释放。实机测试加载
完整 GGUF，在同一模型实例中执行以下检查：

1. 使用 greedy sampling 依次请求输出 2、3、2，验证默认逻辑复位没有会话串扰；
2. 启用 `zero_kv_on_reset=True`，连续两次执行同一请求，检查物理复位后的输出一致性；
3. 输入 256 个 token 检查 Prefill 上限，输入 257 个 token 检查越界保护；
4. 生成 EOG 后再次调用 `decode()`，检查完成状态保护；
5. 检查加载前、加载中和关闭后的 GGUF 文件描述符；
6. 检查 `/proc/self/maps`，确认加载 TCIM Runtime 且未加载 llama.cpp/HLIELLama。

### 3.3 性能测试方法

性能测试加载一次模型，使用固定 28-token 提示词、greedy sampling 和固定
32-token 输出。进程固定在主机 CPU 4–7，先执行 5 次预热，再保留 20 次正式
样本。每次样本均执行完整的 `prefill()` 和 32 次 `decode()` 调用；首 token 来自
Prefill logits，后续 31 个 token 对应 31 次 Decode 计算图执行。

测试使用 `time.perf_counter()` 统计主机墙钟时间。TCIM 图时间包含 `run()` 和
`sync()`；调用方可见首 token 时间包含 Prefill、首 token CPU 采样及 API 返回；
完整请求时间包含 Prefill 和全部 Decode 调用。时延报告 P50 和最近秩 P95，吞吐
报告 20 次样本的中位数。测试期间 M50 保持 1300 MHz、performance 模式。

### 3.4 与 FlashRT 其他后端测试方法的对照

FlashRT 的 CUDA Qwen 后端通常采用两级验证：无模型单元测试加真实 checkpoint
GPU 测试；正确性还会与 Hugging Face BF16 参考实现比较 logits cosine、首 token
argmax 和 greedy token 序列。性能测试通常包含预热、设备同步、多次样本和
P50/P95 或 token/s。

本次 M50 测试已经补齐预热、多样本统计、实机状态和资源检查，但本机没有
Qwen3-0.6B 原始 BF16 checkpoint 或 golden logits，因此尚未完成 HF 数值对齐。
报告中的“正确性”限定为当前 M50 交付件的运行时链路、确定性和状态正确性，
不等同于量化模型精度评估。HLIELLama 未作为被测后端或参考后端运行。

## 4. 正确性测试命令和结果

### 4.1 单元测试

测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_m50_qwen_frontend.py
```

测试结果：

```text
.................                                                        [100%]
17 passed in 0.28s
```

该命令直接运行 M50/Qwen 专项单元测试，不加载模型和 M50。测试使用最小 GGUF
夹具和模拟 TCIM 对象检查解析、接口契约及异常路径。

### 4.2 M50 实机正确性测试

测试命令：

```bash
python examples/m50/qwen_tcim_runtime_check.py \
  --model /home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf \
  --max-tokens 16 \
  --output-json docs/m50/results/qwen3_0.6b_tcim_runtime_check_20260817.json
```

`qwen_tcim_runtime_check.py` 只负责正确性和状态检查，不统计性能。程序加载一次
模型，先在默认逻辑复位模式下执行 2→3→2，再打开物理 KV Cache 清零执行两次
相同请求，随后检查 EOG、256/257-token 边界、动态库映射和模型文件描述符，最后
调用 `close()`。

| 检查项 | 实测结果 | 判定 |
|---|---|---|
| HMM 输入输出契约 | Prefill 256，Context 32768，Vocab 151936，Hidden 1024，KV Cache 56 个 | 通过 |
| 默认逻辑复位会话隔离 | 同一实例依次输出 2、3、2 | 通过 |
| 物理 KV Cache 复位 | 两次 token 序列一致；单元测试确认调用 `set_zero()` | 通过 |
| Prefill 输入上限 | 256 token 正常输出有限 logits | 通过 |
| Prefill 越界保护 | 257 token 被拒绝 | 通过 |
| EOG 状态保护 | EOG 后继续 Decode 被拒绝 | 通过 |
| 动态库映射 | TCIM Runtime 已加载，`libllama.so` 未加载 | 通过 |
| GGUF 文件描述符 | 加载前 0、加载中 1、关闭后 0 | 通过 |
| 测试后设备状态 | device 0 正常，频率 1300 MHz | 通过 |

2→3→2 三次会话的实际 token 序列分别为：

```text
2：[151667, 271, 151668, 271, 17, 151645]
3：[151667, 271, 151668, 271, 18, 151645]
2：[151667, 271, 151668, 271, 17, 151645]
```

## 5. 性能测试命令和结果

测试命令直接调用 Python 性能程序，不经过 Shell 封装脚本：

```bash
taskset -c 4-7 python examples/m50/qwen_tcim.py \
  --model /home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf \
  --prompt '请用中文详细说明矩阵乘法的计算过程，并给出一个例子。 /no_think' \
  --max-tokens 32 \
  --warmup 5 \
  --repeat 20 \
  --output-json docs/m50/results/qwen3_0.6b_tcim_throughput_20260817.json
```

`qwen_tcim.py` 完成以下操作：加载一次模型；统计模型加载时间；执行 5 次不计入
结果的预热；执行 20 次正式 Prefill 和 Decode；分别记录 TCIM 图执行时间、主机
墙钟时间、调用方首 token 时间、完整请求时间和 token 序列；最后计算 P50/P95
并写入 JSON。脚本同时检查进程动态库映射。

测试条件：M50 1300 MHz、2 核、batch 1、greedy sampling、28-token 输入、
32-token 固定输出、主机 CPU 亲和性 4–7、5 次预热、20 次计时。

| 性能指标 | P50 | P95或说明 |
|---|---:|---:|
| 模型加载时间 | 14.237 s | 单次加载，不计算分位数 |
| Prefill 阶段墙钟时延 | 33.799 ms | 34.408 ms |
| Prefill 计算图时延 | 26.343 ms | 26.616 ms |
| 调用方可见首 token 时延 | 36.306 ms | 37.049 ms |
| 32-token 完整请求时延 | 656.131 ms | 680.360 ms |
| Prefill 后生成吞吐 | 51.463 token/s | 20 次样本中位数 |
| 完整请求输出吞吐 | 48.771 token/s | 包含 Prefill |
| Decode 计算图吞吐 | 70.948 step/s | 每次请求 31 step |
| Decode 图外主机时间 | 185.666 ms | 201.268 ms |

在相同 5 次预热、20 次计时但未设置 CPU 亲和性的对照中，Decode 图吞吐为
71.289 step/s，图外主机时间 P50 为 366.297 ms。固定 CPU 4–7 后，M50 图吞吐
基本不变，图外时间降至 185.666 ms，说明吞吐变化主要来自 ARM 主机侧数据搬运、
采样和线程调度，而不是 M50 频率变化。

20 次正式样本的 token 序列完全一致。输出为：

```text
<think>

</think>

矩阵乘法是一种用于将两个矩阵相乘的数学运算，其基本原理是将两个矩阵的元素按照特定的规则相
```

本次设置达到 32-token 上限时尚未生成 EOG，因此文本在“相”处截断，这是固定
长度性能测试的预期行为。原始逐次数据保存在：

- [`results/qwen3_0.6b_tcim_20260817.json`](results/qwen3_0.6b_tcim_20260817.json)
- [`results/qwen3_0.6b_tcim_runtime_check_20260817.json`](results/qwen3_0.6b_tcim_runtime_check_20260817.json)
- [`results/qwen3_0.6b_tcim_throughput_20260817.json`](results/qwen3_0.6b_tcim_throughput_20260817.json)

## 6. 测试结论和未覆盖项

当前 Qwen3-0.6B 交付件已经通过 FlashRT → TCIM → M50 的实机运行时正确性和
单流性能测试。首 token 改为按需 Decode，并将性能进程固定到主机高性能核后，
调用方可见首 token P50 为 36.306 ms；固定 32-token 的 Prefill 后生成吞吐为
51.463 token/s。

测试尚未覆盖以下项目，不能将本报告解释为完整的模型精度或生产验收报告：

- 与 Qwen3-0.6B 原始 BF16/HF 模型的 logits cosine、argmax 和长序列 token 对齐；
- 不同输入长度和接近 32K Context 的性能曲线；
- temperature、top-k 和 top-p 随机采样的统计正确性；
- 多实例并发、长时间稳定性、峰值内存、负载功耗和温度；
- 其他 Qwen 型号或不同 HMM 编译规格的兼容性。

调试过程及已修复问题见
[`qwen_tcim_debug_report_20260817.md`](qwen_tcim_debug_report_20260817.md)。
