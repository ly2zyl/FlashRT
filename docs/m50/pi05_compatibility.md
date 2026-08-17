# Pi0.5 M50 HMM 兼容前端状态说明

最后核对日期：2026-08-17。

## 1. 当前状态

FlashRT 已实现 `Pi05M50Frontend`，用于调度后摩工具链预先生成的 Pi0.5 六图
HMM 交付件。当前本机没有完整且匹配的模型资源包，因此只完成资源校验、API
分发和动作变换单元测试，尚未完成 Pi0.5 模型实机推理。

该前端属于既有 HMM 的兼容运行路径，不应描述为 FlashRT 已从 PyTorch 权重
完成 Pi0.5 的 M50 原生量化编译。

## 2. 执行链路

```text
FlashRT VLAModel API
  -> Pi05M50Frontend：预处理、图调度、后处理
  -> tcim_lite
  -> Pi0.5 HMM 静态图
  -> M50/XH2
```

## 3. 所需资源

策略 checkpoint 目录至少需要：

```text
pi05_checkpoint/
├── config.json
├── policy_preprocessor_step_*_normalizer_processor.safetensors
└── policy_postprocessor_step_*_unnormalizer_processor.safetensors
```

PaliGemma tokenizer 目录至少需要：

```text
paligemma-3b-pt-224/
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```

M50 编译产物目录需要：

```text
pi05/xh2/
├── siglip.hmm
├── gemma_2b_prefill.hmm
├── gemma_expert_300m_decode.hmm
├── action_in_proj.hmm
├── action_out_proj.hmm
├── time_mlp.hmm
└── embedding.pt
```

六个 HMM、embedding、checkpoint 和 tokenizer 必须属于同一模型配置，并与
M50 驱动、固件及 TCIM Runtime 版本匹配。

## 4. 资源检查

```bash
cd /home/sky/icode/FlashRT

/home/sky/icode/.venv-m50-runtime/bin/python examples/m50/pi05_doctor.py \
  --checkpoint /path/to/pi05_checkpoint \
  --compiled-model-dir /path/to/pi05/xh2 \
  --tokenizer /path/to/paligemma-3b-pt-224
```

资源齐全后可执行单次推理：

```bash
/home/sky/icode/.venv-m50-runtime/bin/python examples/m50/pi05_smoke.py \
  --checkpoint /path/to/pi05_checkpoint \
  --compiled-model-dir /path/to/pi05/xh2 \
  --tokenizer /path/to/paligemma-3b-pt-224 \
  --image /path/to/base.png \
  --image /path/to/wrist.png \
  --prompt 'put the bowl on the stove' \
  --state 0 0 0 0 0 0 0 0
```

## 5. Python API

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",
    config="pi05",
    framework="torch",
    hardware="m50_hmm_compat",
    compiled_model_dir="/path/to/pi05/xh2",
    tokenizer_path="/path/to/paligemma-3b-pt-224",
    device_id=0,
)

actions = model.predict(
    images=[base_rgb, wrist_rgb],
    prompt="put the bowl on the stove",
    state=raw_robot_state,
)
```

## 6. 已验证与未验证项目

| 项目 | 状态 |
|---|---|
| `m50_hmm_compat` API 分发 | 单元测试通过 |
| 必需文件和图输入输出校验 | 单元测试通过 |
| 状态归一化与动作反归一化 | 单元测试通过 |
| 六图在 M50 上完整执行 | 未验证，缺少匹配资源包 |
| LIBERO 精度与时延 | 未验证 |
| 其他 embodiment 或不同分图方案 | 未实现 |

在六图资源齐全并完成实机输出、性能和精度验证前，Pi0.5 应保持“兼容前端已
实现，模型部署未完成”的状态。
