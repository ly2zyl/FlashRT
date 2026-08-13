# M50 + FlashRT 可复现环境与部署手册

最后核对：2026-08-13。硬件为后摩 M50/LQ50 M.2、XH2、aarch64。

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
/home/sky/houmo-HLIELLama-xh2
/usr/local/houmo-sdk
/usr/local/houmo
/opt/houmo-tcim-runtime-1.4.0
```

必须修改第三方文件时先复制：

```bash
cp -a /home/sky/houmo-HLIELLama-xh2 /home/sky/icode/houmo-HLIELLama-xh2-local
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
python3.12 -m venv --system-site-packages .venv-m50-runtime
source .venv-m50-runtime/bin/activate
python -m pip install --upgrade pip
```

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
```

查询：

```bash
/home/sky/icode/.venv-m50-runtime/bin/python --version
/home/sky/icode/.venv-m50-runtime/bin/python -m pip freeze
```

该环境使用 `--system-site-packages` 复用了基础环境中的 PyTorch。跨机器时应
先准备同版本 Python/PyTorch；不要把此 venv 当作通用二进制安装包。

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

## 5. FlashRT 版本和构建

```text
源码：/home/sky/icode/FlashRT
分支：codex/m50-qwen-hliellama-handoff
提交：7411d6008183f9e1127d620da8ad041c03a99542
```

构建：

```bash
cd /home/sky/icode/FlashRT
/home/sky/icode/.venv-m50-runtime/bin/cmake -S cpp -B build/houmo-llama \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DFLASHRT_CPP_WITH_EXEC=OFF \
  -DFLASHRT_CPP_WITH_CUDA_STAGING=OFF \
  -DFLASHRT_CPP_WITH_CUDA_KERNELS=OFF \
  -DFLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON \
  -DFLASHRT_CPP_WITH_HOUMO_LLAMA=ON \
  -DHoumoLlama_ROOT=/home/sky/houmo-HLIELLama-xh2
/home/sky/icode/.venv-m50-runtime/bin/cmake --build build/houmo-llama \
  --target flashrt_cpp_llama_cpp_provider_c --parallel 2
```

产物：

```text
/home/sky/icode/FlashRT/build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so
/home/sky/icode/FlashRT/build/houmo-llama/runtime/libflashrt_runtime.so
```

这是 Python API → FlashRT C ABI → 后摩 `libllama.so` → M50 的原生 provider，
不是在 FlashRT 中启动 `llama-cli` 子进程。

## 6. Qwen3.6 测试模型和版本

模型路径（只读）：

```text
/home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf
```

大小为 `21263621984 bytes`（约 19.79 GiB）。元数据为：

```text
architecture=qwen35moe
is_hmm=true
target=xh2
core_num=2
version=v1.4.0
context_length=262144
```

HLIELLama 版本：

```text
llama.cpp=2.1.0
description=lanyue2.1.0
tcim_version=1.3.0
last_update=20260610
```

`convert_hm_to_gguf.py`、`readhmm`、`hmmstrip` 是封装/辅助工具，不是
PyTorch→M50 的量化编译器。

## 7. 后摩 Llama 基线

```bash
export LD_LIBRARY_PATH=/home/sky/houmo-HLIELLama-xh2/lib:/opt/houmo-tcim-runtime-1.4.0/lib
export LLAMA_LOG_VERBOSITY=1
/home/sky/houmo-HLIELLama-xh2/bin/llama-cli \
  --model /home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf \
  --ctx-size 512 --predict 8 --temp 0 \
  --conversation --single-turn --simple-io --no-display-prompt \
  --prompt '请只回答一个数字：一加一等于多少？'
```

本机基线成功，性能约为 prompt `75.9 token/s`、generation `20.7 token/s`。

## 8. FlashRT 测试命令

示例为 `/home/sky/icode/FlashRT/examples/m50/qwen36_hliellama.py`：

```bash
cd /home/sky/icode/FlashRT
export PYTHONPATH=/home/sky/icode/FlashRT
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/home/sky/icode/FlashRT/build/houmo-llama:/home/sky/icode/FlashRT/build/houmo-llama/runtime:/home/sky/houmo-HLIELLama-xh2/lib:/opt/houmo-tcim-runtime-1.4.0/lib
export LLAMA_LOG_VERBOSITY=1
/home/sky/icode/.venv-m50-runtime/bin/python \
  examples/m50/qwen36_hliellama.py \
  --model /home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf \
  --provider-lib /home/sky/icode/FlashRT/build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so \
  --ctx-size 512 --max-tokens 8 \
  --prompt '请只回答一个数字：一加一等于多少？'
```

当前记录：FlashRT provider 已完成 DSO 构建、HAL 初始化、GGUF 元数据读取并
开始向 M50 加载权重；按用户要求在最终生成前停止。因此后摩基线已确认，
FlashRT 最终文本输出需要后续继续等待模型加载后验证。

## 9. 换机器复现检查表

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
/home/sky/houmo-HLIELLama-xh2/lib/libllama.so
对应的 Qwen GGUF
FlashRT/build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so
```

先执行第 7 节基线，再执行第 8 节 FlashRT。首次加载 21 GB 模型可能需要数
分钟，不要同时启动第二个大模型进程。

## 10. Pi0.5 清理和限制

已确认无用的 Pi0.5 专用环境：

```text
/home/sky/icode/.venv-pi05-flashrt
```

该环境已移入回收站，可恢复；没有删除 Pi0.5 源码、配置或其他模型。保留的
M50 环境为 `/home/sky/icode/.venv-m50-runtime`。

本机为 aarch64；若量化编译器只支持 x86_64，应在 x86_64 工具链机器上生成
HMM，再把推理产物复制到 M50 机器。不要把现成 HMM/GGUF 的运行成功描述成
从 PyTorch 源权重完成了 FlashRT 原生量化编译。
