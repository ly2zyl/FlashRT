# M50 + FlashRT 可复现环境与部署手册

最后核对：2026-08-17。硬件为后摩 M50/LQ50 M.2、XH2、aarch64。

## 1. 隔离目录

所有新增代码、虚拟环境、构建目录和日志放在 `/home/sky/icode`：

```text
/home/sky/icode/FlashRT
/home/sky/icode/.venv-m50-runtime
/home/sky/icode/houmo-examples-xh2_v1.4.0/
/home/sky/icode/houmo-Python-xh2_3.0.1_linux_aarch64/
```

以下目录只读使用：

```text
/usr/local/houmo-sdk
/usr/local/houmo
/opt/houmo-tcim-runtime-1.4.0
```

## 2. 机器和配件版本

本机实际版本：

```text
Architecture       aarch64
OS                 Ubuntu 22.04.5 LTS (Jammy)
Kernel             5.10.226
GCC/G++            11.4.0
CMake              3.31.10
Python             3.12.13
HMSW               V1.4.0
HM_SMI             V1.0.0
Driver             V1.4.0
Firmware           V1.4.0
Device             HoumoNPU LQ50-24GB
Memory             24448 MiB
```

检查：

```bash
uname -m
cat /etc/os-release
gcc --version | head -n 1
hm_smi
```

当前驱动和固件已经匹配，不要为部署模型重复刷写固件。固件不提供量化编译器。

## 3. Python 环境

保留环境：`/home/sky/icode/.venv-m50-runtime`。

本机创建命令：

```bash
cd /home/sky/icode
/home/sky/miniforge3/envs/yolo11m-hm/bin/python \
  -m venv --system-site-packages .venv-m50-runtime
source .venv-m50-runtime/bin/activate
python -m pip install --upgrade pip
```

本机 venv 的 `base_prefix` 为只读复用的
`/home/sky/miniforge3/envs/yolo11m-hm`。`pip` 的写入目标是
`/home/sky/icode/.venv-m50-runtime/lib/python3.12/site-packages`，未修改基础 Conda
环境。另一台机器复现时，应准备同版本 Python 3.12 和下列依赖；基础环境名称
不要求相同。

实际包版本：

```text
torch==2.13.0
torchvision==0.28.0
numpy==2.2.6
onnx==1.22.0
onnxruntime==1.28.0
onnxsim==0.7.0
transformers==4.57.6
safetensors==0.8.0
modelscope==1.39.1
modelscope-hub==0.2.0
hmatc==1.4.0.dev0
houmo_tcim_runtime_xh2==1.4.0
pytest==9.1.1
cmake==3.31.10
```

查询：

```bash
/home/sky/icode/.venv-m50-runtime/bin/python --version
/home/sky/icode/.venv-m50-runtime/bin/python -m pip freeze
```

该环境使用 `--system-site-packages` 复用了基础环境中的 PyTorch、HMatC 和
TCIM Python 包。跨机器时应先准备相同架构和版本的依赖；不要把此 venv 当作
可直接复制的通用二进制安装包。

## 4. 后摩环境变量

TCIM Runtime 为 `/opt/houmo-tcim-runtime-1.4.0`：

```bash
source /opt/houmo-tcim-runtime-1.4.0/env.sh
```

官方示例为 `/home/sky/icode/houmo-examples-xh2_v1.4.0/houmo-examples-xh2`：

```bash
cd /home/sky/icode/houmo-examples-xh2_v1.4.0/houmo-examples-xh2
source env.sh
```

关键变量：

```text
HOUMO_SDK_PATH=/usr/local/houmo-sdk
HOUMO_PATH=/usr/local/houmo
TCIM_RUNTIME_PATH=/opt/houmo-tcim-runtime-1.4.0
HOUMO_EXAMPLES_PATH=/home/sky/icode/houmo-examples-xh2_v1.4.0/houmo-examples-xh2
```

本机没有公开 PyPI 可安装的 `xhquant` 和 `tcim.builder`；它们属于后摩授权的
量化/编译工具链，需要官方开发镜像或 wheel。

## 5. FlashRT 版本和原生 TCIM 后端

```text
源码：/home/sky/icode/FlashRT
分支：codex/m50-qwen-tcim-native
```

使用 `git rev-parse HEAD` 记录实际复现实验所用提交。

Qwen M50 路径是 Python FlashRT frontend 直接调用 `tcim_lite`，不依赖
llama.cpp/HLIELLama，不需要构建额外 C++ Provider。安装 FlashRT 本地源码：

```bash
cd /home/sky/icode/FlashRT
/home/sky/icode/.venv-m50-runtime/bin/python -m pip install -e .
```

关键实现：

```text
flash_rt/frontends/m50/gguf_assets.py
flash_rt/frontends/m50/qwen.py
examples/m50/qwen_tcim.py
examples/m50/run_qwen_tcim.sh
```

执行链路为 `FlashRT → tcim_lite → TCIM Runtime → M50`。

## 6. Qwen 测试模型

模型路径（只读）：

```text
/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
```

大小为 `1006385824 bytes`。元数据为：

```text
architecture=qwen3
is_hmm=true
target=xh2
core_num=2
version=v1.2.0
context_length=32768
```

## 7. FlashRT 测试命令

推荐通过环境封装脚本运行：

```bash
cd /home/sky/icode/FlashRT
export MODEL_GGUF=/home/sky/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf
export MAX_TOKENS=32
export REPEAT=3
examples/m50/run_qwen_tcim.sh
```

实机结果为 prefill HMM 26.715 ms、decode 40.485 token/s，且进程未映射
`libllama.so`。详见 `docs/m50_qwen_tcim_native_zh.md`。

## 8. 换机器复现检查表

```bash
uname -m
hm_smi
python3.12 --version
gcc --version
/home/sky/icode/.venv-m50-runtime/bin/python -c 'import tcim_lite, hmatc; print("runtime imports OK")'
```

必须存在：

```text
/usr/local/houmo-sdk/hal/lib/libhal_xh2a.so
/opt/houmo-tcim-runtime-1.4.0/lib/libtcim_runtime_lite.so
包含 `prefill.hmm`、`decoder.hmm`、embedding 和 tokenizer 的 Qwen GGUF
```

测试结果中的 `libllama_mapped` 必须为 `false`，`tcim_runtime_mapped` 必须为
`true`。Tokenizer 小文件会自动缓存到 FlashRT 仓库的 `.cache` 目录。

## 9. Pi0.5 清理和限制

已确认无用的 Pi0.5 专用环境：

```text
/home/sky/icode/.venv-pi05-flashrt
```

该环境已移入回收站，可恢复；没有删除 Pi0.5 源码、配置或其他模型。保留的
M50 环境为 `/home/sky/icode/.venv-m50-runtime`。

本机为 aarch64；若量化编译器只支持 x86_64，应在 x86_64 工具链机器上生成
HMM，再把推理产物复制到 M50 机器。不要把现成 HMM/GGUF 的运行成功描述成
从 PyTorch 源权重完成了 FlashRT 原生量化编译。
