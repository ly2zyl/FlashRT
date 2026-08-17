# 后摩 M50 支持与复现指南

最后核对日期：2026-08-17。

本目录是 FlashRT 后摩 M50/XH2 相关非代码资料的唯一入口。实现代码仍保留在
FlashRT 原有的 `flash_rt/`、`examples/` 和 `tests/` 目录，避免扩大项目结构改动。

## 1. 文档与结果目录

```text
docs/m50/
├── README.md
├── qwen_tcim_debug_report_20260817.md
├── qwen_tcim_test_report_20260817.md
└── results/
    ├── qwen3_0.6b_tcim_20260817.json
    ├── qwen3_0.6b_tcim_throughput_20260817.json
    └── qwen3_0.6b_tcim_runtime_check_20260817.json
```

| 文件 | 内容 | 状态 |
|---|---|---|
| `README.md` | 环境、部署、运行和复现方法 | 当前有效 |
| `qwen_tcim_test_report_20260817.md` | 正确性、性能、测试方法和原始结果 | M50 实机通过 |
| `qwen_tcim_debug_report_20260817.md` | Bug 现象、根因、修复方法和当前状态 | 已完成实机回归 |
| `results/*.json` | 功能与性能测试原始数据 | M50 实机生成 |

## 2. 当前支持范围

Qwen3-0.6B 已完成 M50 实机验证，执行链路如下：

```text
Qwen GGUF 交付件
  -> FlashRT：解析、tokenizer、embedding、prefill/decode 调度、采样、文本解码
  -> tcim_lite
  -> TCIM Runtime：装载并执行 prefill.hmm 和 decoder.hmm
  -> M50/XH2
```

该路径不导入、不链接也不加载 llama.cpp 或 HLIELLama。FlashRT 并不在 M50
机器上重新量化、编译模型；GGUF 中的 HMM 必须由匹配的后摩离线工具链预先生成。

## 3. 测试机器与软件版本

| 项目 | 实际配置 |
|---|---|
| 主机架构 | aarch64 |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 内核 | 5.10.226 |
| Python | 3.12.13 |
| GCC / CMake | 11.4.0 / 3.31.10 |
| M50 | HoumoNPU LQ50-24GB，device 0 |
| HMSW / Driver / Firmware | V1.4.0 / V1.4.0 / V1.4.0 |
| TCIM Runtime | V1.4.0 |
| `houmo_tcim_runtime_xh2` | 1.4.0 |
| `transformers` / `numpy` | 4.57.6 / 2.2.6 |

检查命令：

```bash
uname -m
cat /etc/os-release
hm_smi
/home/sky/icode/.venv-m50-runtime/bin/python --version
```

## 4. 隔离的 Python 环境

本机使用独立虚拟环境：

```text
/home/sky/icode/.venv-m50-runtime
```

本机的创建方式如下。基础环境只读复用，安装操作写入 `icode` 下的新环境：

```bash
cd /home/sky/icode
/home/sky/miniforge3/envs/yolo11m-hm/bin/python \
  -m venv --system-site-packages .venv-m50-runtime
source /home/sky/icode/.venv-m50-runtime/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /home/sky/icode/FlashRT
```

关键 Python 包版本：

```text
torch==2.13.0
torchvision==0.28.0
numpy==2.2.6
transformers==4.57.6
safetensors==0.8.0
hmatc==1.4.0.dev0
houmo_tcim_runtime_xh2==1.4.0
pytest==9.1.1
cmake==3.31.10
```

跨机器复现时，`tcim_lite`、HMatC、HAL、驱动和固件必须使用适配目标架构的
后摩交付版本。这些组件不能用普通 PyPI 包替代，也不建议直接复制本机 venv。

## 5. 运行时与模型资源

本机运行时路径：

```text
/opt/houmo-tcim-runtime-1.4.0
/usr/local/houmo-sdk/hal/lib
```

实测模型只读使用：

```text
/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
```

该文件大小为 1,006,385,824 bytes，目标为 XH2、batch 1、单卡 2 核，编译
上下文长度为 32768，prefill 固定长度为 256。容器内至少应包含：

- `quant_embedding.bin`；
- `prefill.hmm`；
- `decoder.hmm`；
- `tokenizer.json` 及配套 tokenizer 配置。

## 6. 运行 Qwen

推荐使用封装脚本：

```bash
cd /home/sky/icode/FlashRT
export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MAX_TOKENS=32
export REPEAT=3
examples/m50/run_qwen_tcim.sh
```

Python API：

```python
import flash_rt

model = flash_rt.load_model(
    MODEL_GGUF,
    framework="tcim",
    config="qwen",
    device_id=0,
    max_tokens=32,
    temp=0.0,
)

text = model.generate("请只输出数字2，不要输出其他内容。 /no_think")
print(text)
model.close()
```

脚本只设置 TCIM 与 HAL 动态库路径，不设置 HLIELLama 路径。运行结果默认写入
`.cache/m50/results/qwen_tcim.json`；可通过 `RESULT_JSON` 指定其他位置。

## 7. 缓存说明

`.cache/m50/` 只保存以下可再生成内容：

```text
.cache/m50/
├── tokenizer/    # 从 GGUF 提取的 tokenizer 小文件，约 16 MiB
├── results/      # 默认运行结果
└── xdg/          # 后摩运行时生成的用户级日志或状态文件
```

这些文件不是模型权重、HMM 或源码，删除不会影响已提交内容。删除后首次加载会从
GGUF 重新提取 tokenizer，因此启动时间可能略有增加。本次整理完成时已清除缓存，
仓库不携带 `.cache` 内容。

## 8. 复现检查

```bash
cd /home/sky/icode/FlashRT

/home/sky/icode/.venv-m50-runtime/bin/python -c \
  'import tcim_lite; print(tcim_lite.runtime.get_device_num("Xh2HalBackend"))'

/home/sky/icode/.venv-m50-runtime/bin/pytest -q \
  tests/test_m50_qwen_frontend.py
```

Qwen 原生 TCIM 路径是 Python 直接调用 `tcim_lite`，不需要构建 FlashRT C++
扩展。

测试结果见 [`qwen_tcim_test_report_20260817.md`](qwen_tcim_test_report_20260817.md)，
调试修复过程见
[`qwen_tcim_debug_report_20260817.md`](qwen_tcim_debug_report_20260817.md)。

## 9. 已知边界

- 单次 prompt 上限由当前 HMM 固定为 256 token，decode 上下文上限为 32768；
- 当前实现为 batch 1、单会话实例；
- Qwen3.6-35B-A3B 还包含卷积和 SSM 状态，当前原生前端尚未实现对应状态绑定；
- aarch64 M50 机器只执行已有 HMM。若量化编译镜像仅支持 x86_64，应在
  x86_64 工具链机器生成 HMM，再复制到 M50 机器；
- 不应把“加载已有 HMM 成功”描述为“在 M50 上完成 PyTorch 权重量化编译”。
