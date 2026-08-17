# FlashRT 原生 TCIM 后端 M50/Qwen 调试与修复记录

- 调试日期：2026-08-17
- 调试对象：`flash_rt/frontends/m50/qwen.py`
- 当前状态：已修复并完成 M50 实机回归
- 对应提交：`cde645e fix: correct M50 Qwen TCIM runtime state`

## 1. 调试目标

对 FlashRT → `tcim_lite` → TCIM Runtime → M50 链路进行逐段检查，区分
FlashRT 代码缺陷、TCIM Runtime 提示和模型交付件固有限制。本文只记录调试
过程、问题根因、修复措施和当前状态；最终正确性与性能结果见
[`qwen_tcim_test_report_20260817.md`](qwen_tcim_test_report_20260817.md)。

## 2. 问题汇总

| 编号 | 问题 | 影响 | 当前状态 |
|---:|---|---|---|
| 1 | KV Cache 所有权与绑定方向错误 | 不符合 TCIM dummy tensor 契约，存在运行时版本兼容和内存所有权风险 | 已修复 |
| 2 | Decode 热路径重复创建输入数组 | 增加逐 token Python 分配与调度开销 | 已修复 |
| 3 | Greedy 采样重复创建 FP32 logits | 增加内存分配和全词表采样开销 | 已修复 |
| 4 | EOG 后仍允许继续调用 decode | 分阶段 API 的生命周期状态不完整 | 已修复 |
| 5 | 运行参数缺少边界校验 | 非法参数可能延迟到模型加载或采样阶段才失败 | 已修复 |
| 6 | `close()` 未显式释放 embedding memmap | 实例关闭后可能继续占用 GGUF 文件描述符 | 已修复 |
| 7 | TCIM backend warning 被误认为错误 | 容易造成测试结论误判 | 已确认是 Runtime 提示 |
| 8 | Decode 仍有图外 host 时间 | 限制当前端到端吞吐 | 现有 HMM 接口限制，非遗留代码错误 |

## 3. 问题分析与修复

### 3.1 KV Cache 绑定方向

问题现象：prefill 和 decoder 均能加载，但代码从设置了 dummy Cache 输入的
decoder 获取设备 Tensor，再反向绑定给 prefill。

根因：`Option.set_dummy_tensors()` 表示 decoder 不负责为这些输入分配内存。
按照 TCIM 接口和后摩示例，应由 prefill 持有 Cache，再绑定给 decoder。原实现
虽然在当前 Runtime 上能够执行，但所有权方向错误。

修复：

```python
cache = self._prefill.get_dev_input(name)
self._decode.set_input(name, cache)
```

验证：两个 HMM 正常加载；同一实例连续执行 2→3→2，结果分别为 2、3、2；
`zero_kv_on_reset=True` 连续两次输出一致。

当前状态：已修复。

### 3.2 Decode 输入数组重复分配

问题现象：每个 token 都重新创建 embedding、`valid_length` 和
`current_length` NumPy 数组。

根因：固定 shape 的输入没有在模型初始化阶段复用，造成不必要的 Python 对象
分配和数据布局检查。

修复：初始化时预分配以下数组，decode 时只更新内容：

- `_decode_embedding`；
- `_decode_valid_length`；
- `_decode_current_length`；
- `_prefill_embeddings` 及对应长度数组。

同时缓存 prefill、decoder 的输入输出名称，避免每 token 重复跨 Python/C++
边界查询。

当前状态：已修复并通过输出一致性测试。

### 3.3 Greedy 采样临时缓冲区

问题现象：每个 token 都把 151936 项 FP16 logits 转换成新建的 FP32 数组。

根因：采样函数没有复用固定词表大小的工作区。

修复：初始化 `_sampling_logits` FP32 缓冲区，每步使用 `np.copyto()` 更新后执行
argmax；temperature 采样也复用该缓冲区。缓存修改不会影响 TCIM 输出 Tensor。

验证：32-token 端到端中位吞吐从 38.123 提升到 41.427 token/s；token 序列
保持一致。

当前状态：已修复。

### 3.4 生成完成状态

问题现象：生成 EOG 后，调用者仍可继续执行 `decode()`。

根因：原实现只检查生成 token 数量，没有记录 EOG 或预算耗尽状态。

修复：增加 `_finished` 状态；EOG 或达到 `max_tokens` 时置位，`reset()` 和新的
prefill 清除状态，完成后继续 decode 会被拒绝。

验证：新增专项单元测试，首次 decode 返回 EOG，第二次调用被完成状态保护拦截。

当前状态：已修复。

### 3.5 参数边界校验

问题现象：负 `device_id`、`temp`、`top_k` 或超出 `[0, 1]` 的 `top_p` 没有在
模型构造入口统一校验。

修复：在访问模型文件和初始化 TCIM 前完成参数检查。

验证：新增 5 组参数化单元测试。

当前状态：已修复。

### 3.6 GGUF 内存映射释放

问题现象：`close()` 释放 TCIM module 后，embedding 的 NumPy memmap 没有显式
关闭。

根因：只清空了 module 和 Cache 引用，没有关闭 memmap 的底层 `_mmap`。

修复：`close()` 中关闭 mmap，并清空 embedding、采样和输入缓冲区引用。

验证：加载前、加载中和关闭后指向模型 GGUF 的文件描述符数量为 0→1→0。

当前状态：已修复。

## 4. 已核查但不是代码缺陷的项目

### 4.1 TCIM backend warning

运行时会打印：

```text
Using TCIM_BACKEND = Xh2HalBackend as default backend
Empty backend name, use Xh2HalBackend instead.
```

核查结果：后摩官方 YOLO、算子和 `tcim_perf` 工具通过环境变量选择 backend 时
会打印相同信息；当前 `tcim_lite.runtime.Option` 没有 backend setter。模型实际
使用 `Xh2HalBackend` 并完成推理，因此这是 TCIM Runtime 的默认 backend 提示，
不是 FlashRT 异常。

当前状态：无需修改 FlashRT。

### 4.2 Decode 图外时间

当前 decoder HMM 输出 `[1, 1, 151936]` FP16 完整词表 logits。每个 token 需要
向 host 回传约 304 KiB，并完成三个输入绑定和 CPU 采样。代码侧临时分配已
消除，但数据传输和 TCIM API 调用仍然存在。

若需继续明显提速，需要离线编译设备端 sampler/argmax HMM，或由 TCIM 只回传
候选 token。现有交付件没有该图，因此此项属于模型产物与接口边界。

当前状态：不是遗留功能 bug；作为后续性能优化项保留。

## 5. 回归验证

| 验证项 | 结果 |
|---|---|
| Qwen 专项单元测试 | 10/10 通过 |
| M50 短回答重复测试 | 3/3 token 序列一致 |
| 同实例 2→3→2 状态隔离 | 通过 |
| 物理 KV Cache 复位 | 通过 |
| EOG 后 decode 保护 | 通过 |
| GGUF 文件描述符释放 | 0→1→0 |
| `libllama.so` 映射检查 | 未加载 |
| 测试结束后的 M50 状态 | 正常 |

## 6. 当前状态

已发现的 FlashRT 功能和资源管理问题均已修复，并完成 M50 实机回归。当前路径
可用于该 Qwen3-0.6B 交付件的 batch-1 单会话推理。剩余图外耗时需要新的设备端
采样产物或 TCIM 接口能力，不影响当前正确性结论。
