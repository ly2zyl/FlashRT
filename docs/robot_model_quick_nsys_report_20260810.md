# 五模型 LIBERO Quick、频率与 Nsight Systems 对比报告

> 本文件保留早期“四用例 × 1 episode”的性能与 NSys 记录。当前 quick 已升级为六类
> 代表动作、每类 10 episodes；最新质量结果见
> `docs/robot_model_representative_quick_report_20260810.md`。

测试日期：2026-08-10
GPU：NVIDIA GeForce RTX 5090（SM120，32 GB）
驱动：590.44.01
容器：`flashrt-robot-bench`（`flashrt:5090`）
Nsight Systems：2025.5.1.121

## 1. 结论摘要

- 本轮重新完成了五模型 × 四个 LIBERO 测试用例，共 20 个 quick episodes。
- Quick 成功数：VLA-JEPA 4/4、MolmoAct2 4/4、SmolVLA 3/4、Pi0.5 2/4、X-VLA 0/4。
- 单次完整重规划频率：Pi0.5 45.76 Hz、VLA-JEPA 15.72 Hz、X-VLA 12.98 Hz、
  SmolVLA 7.44 Hz、MolmoAct2 6.52 Hz。
- Pi0.5 的 kernel 时间占 replan P50 约 92.5%，且使用 CUDA Graph，调度效率最好。
- SmolVLA 的 GPU kernel 只占 P50 约 21.1%，一次 replan 发射 13,646 个 kernel，主要
  瓶颈是 eager/launch 碎片，而不是 GPU 算力。
- X-VLA 的 GPU 时间占约 70.4%，其中 GEMM/MMA 约 74.0%、attention 约 16.8%，主要是
  FP32 计算瓶颈。
- VLA-JEPA 的 GPU kernel 仅占约 31.8%，Memcpy、同步和框架调度占比较大。
- MolmoAct2 一次 replan 发射 35,080 个 kernel，P95/P50 达 1.86，吞吐和尾延迟均最弱。

本轮测试时 GPU 不是独占状态。测试完成后的快照仍显示两个外部计算进程共占约
3.96 GiB，GPU utilization 47%、功耗约 229.5 W。本轮 P50 比此前归档慢 13%–35%，
所以本报告中的绝对频率是“共享 GPU 条件下的当次实测”，不能替代空闲 GPU 上的最终
基准。

## 2. Task、test case 和 episode

`task/test case` 是任务定义；`episode` 是环境 reset 后对该任务的一次完整尝试，直到成功
或超时。Quick 选择四个 test case，每个 test case 只运行 1 个 episode：

| 列名 | Suite / task | 测试内容 | 超时步数 |
|---|---|---|---:|
| Spatial | `libero_spatial / 0` | 把盘子和 ramekin 之间的黑碗放到盘子上 | 280 |
| Object | `libero_object / 0` | 把 alphabet soup 放进篮子 | 280 |
| Goal | `libero_goal / 0` | 打开柜子中间抽屉 | 300 |
| Long | `libero_10 / 0` | 把 alphabet soup 和 tomato sauce 都放进篮子 | 520 |

所有 episode 使用 seed 0、双相机、相同 LIBERO 环境和相对控制模式。

## 3. Quick 主对比表

表中控制频率为 `episode 步数 / episode 墙钟时间`，包含预处理、策略动作选择和 MuJoCo
environment step。括号中为实际执行步数。

| 模型 | Spatial（1 episode） | Object（1 episode） | Goal（1 episode） | Long（1 episode） | 合计 | 聚合控制频率（Hz） |
|---|---|---|---|---|---:|---:|
| Pi0.5 | 成功 · 43.75 Hz（152） | 失败 · 70.27 Hz（280） | 成功 · 72.92 Hz（167） | 失败 · 56.96 Hz（520） | 2/4 | 59.27 |
| SmolVLA | 成功 · 26.84 Hz（80） | 成功 · 44.50 Hz（127） | 成功 · 51.76 Hz（126） | 失败 · 68.03 Hz（520） | 3/4 | 53.60 |
| X-VLA | 失败 · 34.64 Hz（280） | 失败 · 50.75 Hz（280） | 失败 · 66.99 Hz（300） | 失败 · 63.22 Hz（520） | 0/4 | 52.46 |
| VLA-JEPA | 成功 · 19.34 Hz（75） | 成功 · 38.41 Hz（136） | 成功 · 35.51 Hz（121） | 成功 · 37.16 Hz（264） | 4/4 | 33.24 |
| MolmoAct2 | 成功 · 18.20 Hz（82） | 成功 · 22.33 Hz（140） | 成功 · 20.51 Hz（118） | 成功 · 17.85 Hz（253） | 4/4 | 19.31 |

聚合控制频率用四个 episode 的总步数除以总墙钟计算，不是四个频率的算术平均。

### 如何理解这张表

- 控制频率高不代表任务质量高。X-VLA 的控制循环很快，但四次均运行到超时。
- 失败 episode 通常执行到最大步数，所以频率可能比提前成功的 episode 更稳定或更高。
- 策略会生成 action chunk，并在后续环境步复用缓存动作；因此控制频率可以明显高于模型
  的重规划频率。模型性能比较应优先看下一节的重规划频率。
- 每个 test case 只有一个 episode，成功率的分母太小。本表只能验证端到端链路并暴露
  明显问题，不能作为最终质量排名。

## 4. 模型性能：以重规划频率为主

一次 replan 从相同的两路 CPU 图像、文本和状态开始，包含 resize、tokenization、
CPU/GPU 搬运和完整策略推理，最终统一返回第一步 7 维动作。数据为 warmup 10 次后正式
测量 30 次。

| 模型 | 重规划频率（Hz，P50） | P50 | P95 | P95/P50 | 峰值显存 | 冷加载 | 首次推理 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pi0.5 | **45.76 Hz** | 21.86 ms | 22.46 ms | 1.03 | 9.03 GiB | 6.03 s | 532 ms |
| SmolVLA | 7.44 Hz | 134.36 ms | 144.36 ms | 1.07 | **0.90 GiB** | 8.42 s | 665 ms |
| X-VLA | 12.98 Hz | 77.03 ms | 82.20 ms | 1.07 | 3.44 GiB | 13.05 s | 629 ms |
| VLA-JEPA | 15.72 Hz | 63.61 ms | 66.53 ms | 1.05 | 5.86 GiB | 13.04 s | 612 ms |
| MolmoAct2 | 6.52 Hz | 153.31 ms | 284.62 ms | **1.86** | 11.10 GiB | 14.70 s | 1,527 ms |

### 性能排序

1. Pi0.5：45.76 Hz，约为第二名 VLA-JEPA 的 2.91 倍。
2. VLA-JEPA：15.72 Hz。
3. X-VLA：12.98 Hz。
4. SmolVLA：7.44 Hz。
5. MolmoAct2：6.52 Hz，且尾延迟波动最大。

SmolVLA 的优势是显存，仅约 0.90 GiB；它不是速度最快的模型。MolmoAct2 的 P95 比
P50 高 86%，说明当前路径存在严重尾延迟问题。

## 5. Nsight Systems 总览

NSys 对每个模型捕获一次相同输入边界的完整 replan。这里分析的是模型策略路径，不包含
MuJoCo environment step。`kernel/P50` 和“近似非 kernel”由两次同环境但独立运行的
测量组合得出，只用于判断瓶颈方向。

| 模型 | GPU kernel 总时间 | Kernel 数 | 平均 kernel | Kernel/P50 | 近似非 kernel | CUDA Graph |
|---|---:|---:|---:|---:|---:|---|
| Pi0.5 | 20.215 ms | 2,735 | 7.39 μs | **92.5%** | 1.640 ms | 是 |
| SmolVLA | 28.353 ms | 13,646 | **2.08 μs** | 21.1% | **106.012 ms** | 否 |
| X-VLA | 54.237 ms | 3,540 | 15.32 μs | 70.4% | 22.788 ms | 否 |
| VLA-JEPA | 20.256 ms | 4,588 | 4.42 μs | 31.8% | 43.350 ms | 否 |
| MolmoAct2 | **134.548 ms** | **35,080** | 3.84 μs | 87.8% | 18.762 ms | 部分路径 |

### GPU kernel 类型占比

| 模型 | GEMM/MMA | Attention | Elementwise/Norm/Reduce | Memory transform | 其他 |
|---|---:|---:|---:|---:|---:|
| Pi0.5 | 66.1% | 16.8% | 12.2% | 0.0% | 4.9% |
| SmolVLA | 41.0% | 2.6% | **54.8%** | 1.4% | 0.3% |
| X-VLA | **74.0%** | 16.8% | 8.3% | 0.3% | 0.6% |
| VLA-JEPA | 62.4% | 5.8% | 26.3% | 2.5% | 3.0% |
| MolmoAct2 | 36.7% | **20.8%** | 40.3% | 2.2% | 0.0% |

### CUDA API 行为

单元格格式为“调用次数 / API 墙钟”。同步 API 的时间通常是在等待 GPU，不能与 GPU
kernel 时间直接相加。

| 模型 | Kernel launch | Graph launch | Synchronize | Memcpy |
|---|---:|---:|---:|---:|
| Pi0.5 | 2 / 0.089 ms | 1 / 5.754 ms | 6 / 17.388 ms | 6 / 0.216 ms |
| SmolVLA | **13,646 / 40.338 ms** | 0 | 37 / 0.877 ms | 524 / 3.231 ms |
| X-VLA | 3,540 / 10.450 ms | 0 | 53 / 1.896 ms | 77 / 0.651 ms |
| VLA-JEPA | 4,588 / 14.157 ms | 0 | **117 / 12.192 ms** | **126 / 12.585 ms** |
| MolmoAct2 | 3,920 / 11.657 ms | 1 / 36.160 ms | 19 / **83.642 ms** | 119 / 1.400 ms |

## 6. 逐模型 NSys 分析

### Pi0.5

- GPU kernel 已占 P50 的约 92.5%，近似非 kernel 开销只有 1.64 ms。
- 一个 CUDA Graph launch 覆盖主要推理路径，外部只看到两个直接 kernel launch。
- FP8 MMA 占 kernel 时间 66.1%；三个最热 MMA kernel 分别占 26.7%、17.3%、14.9%。
- FlashAttention split-K 主 kernel 占 11.8%。
- 当前优化重点已经从 Python launch 转向 MMA、attention 和内存布局本身。
- 加载时反复出现 13–18 个 FP8 calibration scale 超过中位数 20 倍的告警。它不阻止
  执行，但需要扩大真实 LIBERO 校准样本，并对离群层做 FP16 fallback 回归。

### SmolVLA

- GPU kernel 总时间只有 28.35 ms，但 replan P50 为 134.36 ms；约 79% 时间不在 GPU
  kernel 中。
- 一次 replan 发射 13,646 个 kernel，平均 kernel 仅 2.08 μs；CPU kernel-launch API
  累计 40.34 ms，没有 CUDA Graph。
- Elementwise、norm 和 reduce 占 kernel 时间 54.8%，超过 GEMM/MMA 的 41.0%。
- 热点包含大量 BF16 小 WMMA、copy 和 elementwise multiply，说明 shape 小、算子碎片化。
- 优先级：固定 shape CUDA Graph、`torch.compile`/算子融合、合并 elementwise、减少 flow
  steps；只优化单个 GEMM 收益有限。

### X-VLA

- GPU kernel 占 P50 约 70.4%，计算瓶颈比框架瓶颈更明显。
- GEMM/MMA 占 74.0%，FP32 memory-efficient attention 占 16.8%。前三个 GEMM 热点合计
  67.8%。
- 一次 replan 发射 3,540 个 kernel，无 CUDA Graph，但平均 kernel 15.32 μs，碎片程度
  明显低于 SmolVLA。
- 优先级：验证 BF16/FP8 精度、优化 FP32 GEMM 和 attention；图捕获可作为第二阶段。
- Quick 四个 test case 均失败。相同单例此前用官方 `lerobot-eval` 交叉检查也失败，
  但一个 episode 仍不足以判定正式质量。

### VLA-JEPA

- GPU kernel 仅 20.26 ms，与 Pi0.5 接近，但 replan P50 达 63.61 ms；约 43.35 ms 是
  非 kernel 开销。
- CUDA API 中 Memcpy 累计 12.59 ms、117 次同步累计 12.19 ms、launch 累计 14.16 ms。
- GEMM/MMA 占 62.4%，elementwise/norm/reduce 占 26.3%，kernel 平均 4.42 μs。
- 优先级：减少 host/device copy 和显式同步，固定 shape 后做 CUDA Graph，再处理小 GEMM
  和 world-model 分支。
- 加载日志存在一个 language-model embedding 的 unexpected key，需要做权重键一致性检查；
  本轮输出有限且四个 quick 用例均成功，尚未观察到运行错误。

### MolmoAct2

- GPU kernel 总时间 134.55 ms、kernel 数 35,080，都是五模型最高。
- Attention 占 20.8%，GEMM/MMA 36.7%，elementwise/norm/reduce 40.3%；不是单一算子
  瓶颈，而是大模型、八次 flow 和大量碎片共同造成。
- 最热 BF16 attention kernel 占 17.9%；一个 elementwise multiply kernel族就有 5,376
  个实例、占 9.7%。
- 存在一个 CUDA Graph launch，但仍有 3,920 个直接 launch，说明只捕获了部分路径。
- P95 284.62 ms，远高于 P50 153.31 ms。优先级是稳定图捕获边界、减少 flow-loop
  发射、合并 elementwise，并调查共享 GPU 竞争下的尾延迟。

## 7. 结果有效性和限制

1. Quick 每个 test case 只有一个 episode，成功率不具备统计稳定性。
2. 本轮 GPU 被其他计算进程共享。相比之前的归档，本轮 P50 变慢：Pi0.5 13.3%、
   SmolVLA 19.2%、X-VLA 19.4%、VLA-JEPA 34.7%、MolmoAct2 22.1%。不同模型对竞争的
   敏感度不同，因此绝对频率和相对差距都应在独占 GPU 后复验。
3. NSys 使用统一的双视角 replan 输入捕获模型策略路径；Quick 使用真实 LIBERO 观测。
   两者 shape 和应用边界对齐，但图像内容和任务文本不同。
4. 控制频率包含环境且受 action chunk 缓存影响；重规划频率不包含环境。两者用途不同，
   不能互相替代。
5. NSys 的 CUDA API 时间可能包含等待和重叠，不与 GPU kernel 总时间直接相加。

## 8. 建议的下一步

1. 等 GPU 无其他计算进程时重新运行性能和 NSys，形成可发布的独占 GPU 基准。
2. 先运行每模型 40 episodes（四 suite × 十任务 × 一 episode）作为中等规模筛查，再对
   候选模型运行标准 400 episodes 和多个 seed。
3. 优化优先顺序：SmolVLA/VLA-JEPA 先解决调度和图捕获；X-VLA 先解决 FP32
   GEMM/attention；MolmoAct2 同时减少 flow-loop 发射和大算子成本；Pi0.5 做精度校准
   回归后再优化 GPU kernel。

## 9. 复现命令和结果位置

```bash
cd /home/zhangyunli/FlashRT

# 本轮 quick
scripts/robot_bench_e2e.sh quick \
  /workspace/models/robot-bench-results/e2e_20260810

# 本轮重规划频率、P50/P95 和显存
scripts/robot_bench_latency.sh all \
  /workspace/models/robot-bench-results/latency_20260810

# 本轮只采集 Nsight Systems（不运行 NCU）
scripts/robot_bench_nsight.sh all \
  /workspace/models/robot-bench-results/nsys_20260810 systems
```

容器内结果：

```text
/workspace/models/robot-bench-results/
├── e2e_20260810/quick/<model>.json
├── latency_20260810/<model>/latency.json
└── nsys_20260810/<model>/
    ├── nsys.nsys-rep
    ├── nsys.sqlite
    ├── nsys_run.json
    ├── nsys_stats_cuda_gpu_kern_sum.csv
    └── nsys_stats_cuda_api_sum.csv
```

宿主机归档：

```text
/home/zhangyunli/FlashRT-profile-artifacts-20260804/
└── robot_profile/libero_quick_nsys_20260810/
```
