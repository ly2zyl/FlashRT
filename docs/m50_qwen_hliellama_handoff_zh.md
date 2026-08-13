# FlashRT 在后摩 M50 上部署 Qwen：进展与续作说明

> 最后更新：2026-08-13（Asia/Shanghai）
>
> 状态：**后摩 HLIELLama 基线推理已成功；FlashRT 原生 Houmo Llama
> provider 已实现并构建，但首次 FlashRT 端到端运行在加载权重期间按用户要求
> 主动停止，尚未得到 FlashRT 路径的最终文本输出。**

本文是给后续 AI/开发者的交接文档。继续工作前请完整阅读，避免重新调查或将
“调用外部 CLI”误当作 FlashRT 集成。

## 1. 用户约束

- 所有新增代码、构建产物、日志和可修改文件必须放在
  `/home/sky/icode`。
- `/home/sky/houmo-HLIELLama-xh2`、`/home/sky/qwen3-asr-*` 等其他人的
  资源只允许读取，禁止修改。必须修改模型或第三方资源时，先复制到
  `/home/sky/icode`。
- Python 与构建操作使用 `/home/sky/icode` 下的独立虚拟环境，不修改现有
  Conda 环境或系统 Python。
- 目标是由 FlashRT 的 API 和稳定模型运行时 ABI 驱动 M50，而不是在 FlashRT
  Python 代码里简单执行 `llama-cli` 子进程。

当前使用的隔离环境：

```text
/home/sky/icode/.venv-m50-runtime
```

## 2. 硬件与软件状态

主机架构为 `aarch64`，M50 可正常识别：

```text
HMSW          V1.4.0
Driver        V1.4.0
Firmware      V1.4.0
Device        HoumoNPU LQ50-24GB
Device memory 24448 MiB
```

后摩 HLIELLama 安装目录（只读使用）：

```text
/home/sky/houmo-HLIELLama-xh2
```

其中包含后摩版 `llama-cli`、`libllama.so`、TCIM runtime 和一个完整的
Qwen3.6 M50 量化模型。

## 3. 选定的测试模型

模型为 Qwen3.6-35B-A3B、W4A8、单卡双核、M50/XH2 的 Houmo GGUF：

```text
/home/sky/houmo-HLIELLama-xh2/models/
qwen3.6_35b-a3b_w4a8_262144_1_1/
HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf
```

文件约 21.26 GB。GGUF 元数据说明它内部是后摩 HMM 模型容器：

```text
architecture = qwen35moe
is_hmm       = true
target       = xh2
core_num     = 2
version      = v1.4.0
```

同目录还有视觉投影 GGUF，但当前先做纯文本 LLM，不使用 `mmproj`。

### 关于“Llama 中是否有 Qwen 量化器”

该目录有 `convert_hm_to_gguf.py`、`readhmm`、`hmmstrip` 和 GGUF Python
工具。这些用于把后摩已有 HMM 整理/封装成 GGUF，**不是从 PyTorch/HF
权重生成 M50 HMM 的 `xhquant` 量化器或 `tcim.builder` 编译器**。

当前测试模型已经由后摩量化编译完成，因此运行推理不需要 `xhquant`。

## 4. 已验证的基线

后摩原生 `llama-cli` 已在 M50 上成功加载同一 GGUF，并完成中文生成：

```bash
export LD_LIBRARY_PATH=/home/sky/houmo-HLIELLama-xh2/lib:/opt/houmo-tcim-runtime-1.4.0/lib
export LLAMA_LOG_VERBOSITY=1

/home/sky/houmo-HLIELLama-xh2/bin/llama-cli \
  --model /home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf \
  --ctx-size 512 \
  --predict 8 \
  --temp 0 \
  --conversation --single-turn --simple-io --no-display-prompt \
  --prompt '请只回答一个数字：一加一等于多少？'
```

实际结果包含有效生成，性能日志为：

```text
Prompt     约 75.9 token/s
Generation 约 20.7 token/s
```

因此模型、后摩 Llama 运行时、M50 驱动/固件链路均已验证可用。

## 5. FlashRT 已实现的原生接入

新增 C++ engine：

```text
cpp/providers/llama_cpp/src/houmo_llama_engine.cpp
```

它直接链接后摩 `libllama.so`，将文本 LLM 映射到 FlashRT 已有的
`frt_llama_cpp_engine_v1`，再通过 `llm_runtime.cpp` 暴露稳定的
`frt_model_runtime_v1`：

```text
flash_rt.load_model(...)
  -> LlmHoumoFrontend
  -> libflashrt_cpp_llama_cpp_provider_c.so
  -> frt_model_runtime_v1
  -> Houmo libllama.so
  -> HoumoNPU / M50
```

新增 Python frontend：

```text
flash_rt/frontends/houmo_llama/llm.py
```

新增示例：

```text
examples/m50/qwen36_hliellama.py
```

公共调用方式：

```python
import flash_rt

model = flash_rt.load_model(
    MODEL_GGUF,
    framework="houmo_llama",
    config="llm",
    backend="houmo",
    n_ctx=512,
    temp=0.0,
    top_k=0,
    top_p=0.0,
    max_tokens=8,
    lib_path=PROVIDER_DSO,
)
print(model.generate("请只回答一个数字：一加一等于多少？"))
model.close()
```

当前 engine 为整句 `infer`，没有声明 staged-decode 能力；FlashRT 发布单一
`infer` OPAQUE stage。后续如需 `reset -> prefill -> decode`，必须实现并验证
真实的分阶段状态，不能用缓存结果伪装。

## 6. 构建方法

构建目录全部位于 FlashRT 仓库内：

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

构建已成功，产物为：

```text
/home/sky/icode/FlashRT/build/houmo-llama/
libflashrt_cpp_llama_cpp_provider_c.so
```

以及：

```text
/home/sky/icode/FlashRT/build/houmo-llama/runtime/libflashrt_runtime.so
```

## 7. 端到端运行命令

后摩目录自带的 `libstdc++.so.6` 较旧，缺少 `GLIBCXX_3.4.29`。不要替换或
删除它。运行 FlashRT provider 时让系统 C++ runtime 排在后摩目录之前：

```bash
cd /home/sky/icode/FlashRT

export PYTHONPATH=/home/sky/icode/FlashRT
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/home/sky/icode/FlashRT/build/houmo-llama:/home/sky/icode/FlashRT/build/houmo-llama/runtime:/home/sky/houmo-HLIELLama-xh2/lib:/opt/houmo-tcim-runtime-1.4.0/lib
export LLAMA_LOG_VERBOSITY=1

/home/sky/icode/.venv-m50-runtime/bin/python \
  examples/m50/qwen36_hliellama.py \
  --model /home/sky/houmo-HLIELLama-xh2/models/qwen3.6_35b-a3b_w4a8_262144_1_1/HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf \
  --provider-lib /home/sky/icode/FlashRT/build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so \
  --ctx-size 512 \
  --max-tokens 8 \
  --prompt '请只回答一个数字：一加一等于多少？'
```

## 8. 当前停止位置与未完成事项

上面的 FlashRT 命令已经成功完成以下阶段：

1. Python 导入 FlashRT；
2. `load_model(framework="houmo_llama")` 进入新 frontend；
3. `ctypes` 加载 FlashRT provider DSO；
4. provider 通过 `frt_model_runtime_v1` 创建 Houmo engine；
5. 后摩 `libllama.so` 成功加载；
6. 后摩 HAL 成功识别 M50、V1.4.0 驱动和 24448 MiB 内存；
7. 成功读取 Qwen3.6 GGUF 元数据并开始向设备加载模型权重。

在权重加载过程中，用户要求记录进展并停止，因此发送了 Ctrl-C。随后的
`Interrupted system call`、`COPY_BUFFER failed`、`LoadWeights failed` 是主动中断
正在进行的 H2D 权重拷贝造成的，不是此前发生的独立功能故障。中断后
`hm_smi` 仍正常识别设备。

尚未完成：

- FlashRT 路径尚未输出最终生成文本；
- `houmo_llama_engine.cpp` 尚未经过第二轮运行与边界情况测试；
- 尚未添加 mock 单元测试或 provider export 测试；
- 尚未验证同一进程连续调用两次 `generate()`；
- 尚未比较 FlashRT 输出与 `llama-cli` 的贪心输出；
- 尚未测量 FlashRT 路径 token/s；
- engine 当前会自动应用 GGUF chat template，需要验证模板输出是否与
  `llama-cli --conversation --single-turn` 一致；
- FlashRT provider 构建依赖外部只读 HLIELLama 前缀，尚未做可搬运安装包。

因此后续工作应从“重新运行第 7 节命令并等待生成完成”开始，不要重新下载
模型，也不要重新实现 shell 子进程 wrapper。

## 9. Pi0.5 与其他本机资源

用户已决定暂不推理 Pi0.5。唯一未完成下载的 Pi0.5 权重：

```text
/home/sky/icode/robot_assistant_release_20260509/models/
pi05_libero_finetuned/model.safetensors.incomplete
```

大小约 232.8 MB，已移动到系统回收站；Pi0.5 的代码、配置、tokenizer 和
归一化文件均保留。该操作未触碰 `/home/sky/icode` 之外的模型资源。

本机另有一套完整的 Qwen3-ASR-0.6B M50 HMM，可作为备选，但本次选择了
已有 HLIELLama + Qwen3.6 GGUF，因为它能直接复用 FlashRT 的文本 LLM
provider ABI：

```text
/home/sky/qwen3-asr-models/qwen3_asr_0.6b_1chip_2cores/
```

该目录同样只读使用。

## 10. 安全注意事项

- 不要修改 `/home/sky/houmo-HLIELLama-xh2`；若需修改，先复制到
  `/home/sky/icode`。
- 不要向系统或他人的 Conda 环境安装依赖。
- 不要运行固件升级；当前驱动和固件已是 V1.4.0。
- 不要从公开 PyPI 安装同名 `tcim`，它不是后摩编译器。
- 不要把 HLIELLama 目录中的转换脚本误称为 M50 源模型量化工具。
- 21 GB 模型首次加载较慢，中途无输出不代表卡死。继续测试时至少等待模型
  加载完成，并用 `hm_smi` 辅助判断设备状态。
