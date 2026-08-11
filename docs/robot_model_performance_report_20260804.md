# 五种 LIBERO 机器人策略性能与 Nsight 分析报告

初始测试：2026-08-04；闭环更新：2026-08-10
平台：NVIDIA GeForce RTX 5090（SM120，32 GB），驱动 590.44.01，功耗上限 575 W
容器：`flashrt-robot-bench`，镜像 `flashrt:5090`

> 本文件保留 2026-08-04 的性能基线。当前六类代表动作 × 10 episodes 的质量结果见
> `docs/robot_model_representative_quick_report_20260810.md`；当前频率和 NSys 分析见
> `docs/robot_model_quick_nsys_report_20260810.md`。运行条件不同，不应直接混排行。

## 1. 结论摘要

最终只比较 Pi0.5、SmolVLA、X-VLA、VLA-JEPA 和 MolmoAct2。五个检查点都面向
LIBERO，能够在相同的四个 suite、双相机观测和 7 维动作空间中真实闭环运行。GR00T、
ACT、Diffusion Policy、VQ-BeT、PushT 和独立 Qwen 已从正式实验、权重和结果中清理。

统一推理性能的稳态 P50 排序为 Pi0.5 19.28 ms、VLA-JEPA 47.21 ms、X-VLA
64.52 ms、SmolVLA 112.72 ms、MolmoAct2 125.55 ms。Pi0.5 是本机最快的 replan 路径；
SmolVLA 显存最低；MolmoAct2 的显存和 GPU kernel 总时间最高。

已经完成五模型 × 四 suite 的 20-episode 闭环冒烟测试，证明所有模型都能走通环境观测
到环境动作的端到端链路。这组样本每模型仅四个 episode，不能作为正式成功率排名。
正式质量结论应运行每模型 400 episodes 的 `full` 协议，并最好增加多个 seed。

## 2. 对齐协议

### 性能与 Nsight 协议

- 两路 `float32 [1,3,224,224]` CPU 图像，seed 0；图像对 SHA256 为
  `4a5950d0ca8f2eeb20379c79e6b3b31972a3c6d09cdd44f66b901b129209a9e6`；
- 语言指令固定为 `pick up the red block`；
- 支持状态输入时使用零状态并按检查点需要映射；
- 计时包含 resize、tokenization、CPU/GPU 搬运和完整 replan；
- 统一截取原生 chunk 的第一步 7 维 LIBERO 动作，输出 shape 为 `(1,7)`；
- warmup 10 次、正式测量 30 次，同时记录墙钟和 CUDA Event；
- NSys 捕获一次完整 replan，NCU 对一个代表性 kernel 使用 full metric set。

磁盘图像解码和环境 step 不在推理性能计时区。各模型的 action chunk、精度和推理步数
不同，所以结果表示可部署策略的 replan 成本，而不是等 FLOP 的 runtime 加速比。

### 闭环协议

- 相同 LIBERO MuJoCo 环境和 pinned simulation assets；
- 四组标准 suite：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`；
- 原始双相机环境图像、8 维机器人状态和相同任务文本；
- `control_mode=relative`，经模型和环境各自官方处理器转换后提交 7 维动作；
- `quick` 为每 suite 的 task 0、一个 episode；`full` 为四 suite 的全部 10 个 task、
  每 task 10 episodes。

## 3. 模型配置

| 模型 | 本轮检查点 | 原生计算输出 | 主要精度 |
|---|---|---|---|
| Pi0.5 | `pi05_libero_finetuned_v044` | `10×7`，10 次去噪 | FlashRT FP8 |
| SmolVLA | `lerobot/smolvla_libero@31d453f7` | `1×50×7`，10 次 flow | BF16/FP32 |
| X-VLA | `lerobot/xvla-libero@12e8783e` | `1×30×20`，10 次 flow | FP32 |
| VLA-JEPA | `lerobot/VLA-JEPA-LIBERO@735d9f69` | `1×7×7`，4 次推理 | BF16/FP32 |
| MolmoAct2 | `allenai/MolmoAct2-LIBERO-LeRobot@f0c5a956` | `1×10×7`，8 次 flow | BF16 |

VLA-JEPA 的官方架构内部依赖 Qwen3-VL-2B backbone。它只是 VLA-JEPA 的组成部分，
不是第六个独立对比模型。

## 4. 统一延迟、显存和输出

| 指标 | Pi0.5 | SmolVLA | X-VLA | VLA-JEPA | MolmoAct2 |
|---|---:|---:|---:|---:|---:|
| 参数量 | 约 3.3B | 450.05M | 879.48M | 2.770B | 5.442B |
| 原生 chunk | `10×7` | `1×50×7` | `1×30×20` | `1×7×7` | `1×10×7` |
| 对齐输出 | `(1,7)` | `(1,7)` | `(1,7)` | `(1,7)` | `(1,7)` |
| 墙钟 P50 | 19.28 ms | 112.72 ms | 64.52 ms | 47.21 ms | 125.55 ms |
| 墙钟 P95 | 19.33 ms | 115.69 ms | 65.87 ms | 49.28 ms | 131.83 ms |
| CUDA Event P50 | 19.27 ms | 112.70 ms | 64.51 ms | 47.19 ms | 125.51 ms |
| replans/s（P50） | 51.86 | 8.87 | 15.50 | 21.18 | 7.97 |
| 峰值 allocated | 9.03 GiB | 0.90 GiB | 3.44 GiB | 5.86 GiB | 11.10 GiB |

Pi0.5 的 P95 与 P50 很接近。VLA-JEPA 虽然参数量大于 X-VLA，但只做四次推理，
本机 replan 更快。SmolVLA 的低显存没有转化为低延迟，主要原因是长 action chunk、
10 次 flow 和大量 eager 小算子。MolmoAct2 的模型规模、显存和 P50 都是主组最大。

## 5. Nsight Systems：完整 replan

| 指标 | Pi0.5 | SmolVLA | X-VLA | VLA-JEPA | MolmoAct2 |
|---|---:|---:|---:|---:|---:|
| GPU kernel 总时间 | 17.462 ms | 28.161 ms | 54.152 ms | 19.442 ms | 102.857 ms |
| kernel 实例数 | 2,735 | 13,646 | 3,540 | 4,588 | 35,080 |
| 墙钟 P50 - kernel sum | 1.82 ms | 84.56 ms | 10.37 ms | 27.77 ms | 22.69 ms |

Pi0.5 的 kernel sum 已接近墙钟，FlashRT FP8 路径的调度间隙较小。SmolVLA 的 GPU
kernel 只占约四分之一墙钟，却发射 13,646 个 kernel，瓶颈主要是 flow loop 中的小
GEMM、copy、elementwise 和 Python/eager launch 间隙。VLA-JEPA 也有明显调度间隙。
X-VLA 的 GPU 计算占主导。MolmoAct2 发射 35,080 个 kernel，既有较大计算量，也有
明显的 flow loop 发射碎片。

## 6. Nsight Compute：代表 kernel

| 模型/代表 kernel | Compute SM | Memory / DRAM | occupancy（实测/理论） | waves/SM | 结论 |
|---|---:|---:|---:|---:|---|
| Pi0.5 FP8 MMA 128×128 | 79.76% | 66.69% / 20.92% | 16.19% / 16.67% | 6.02 | Tensor Core 利用率高，waves 充足 |
| SmolVLA BF16 implicit conv | 38.15% | 56.08% / 1.81% | 18.71% / 66.67% | 0.28 | 小 grid、低并行度，非 DRAM 瓶颈 |
| X-VLA TF32 implicit GEMM conv | 24.68% | 38.37% / 4.24% | 18.63% / 33.33% | 0.58 | 代表视觉 kernel 利用率偏低 |
| VLA-JEPA BF16 implicit GEMM conv | 8.85% | 8.78% / 1.30% | 8.34% / 8.33% | 0.19 | 小 grid 且资源受限 |
| MolmoAct2 BF16 CUTLASS GEMM | 20.19% | 32.85% / 5.18% | 16.40% / 16.67% | 1.69 | 非 DRAM 瓶颈，完整路径发射数更关键 |

NCU 的 instrumented duration 受 kernel replay 影响，不能替代端到端延迟。每个模型只抽取
一个可复现的代表 kernel，优化优先级仍应以 NSys 的全路径占比、发射次数和墙钟为准。

## 7. 闭环 quick 结果

| 模型 | Spatial | Object | Goal | Long (`libero_10`) | 合计 | 平均 episode 时间 |
|---|---:|---:|---:|---:|---:|---:|
| Pi0.5 | 0/1 | 0/1 | 1/1 | 0/1 | 1/4 | 3.628 s |
| SmolVLA | 1/1 | 1/1 | 1/1 | 0/1 | 3/4 | 2.688 s |
| X-VLA | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 4.452 s |
| VLA-JEPA | 1/1 | 1/1 | 1/1 | 1/1 | 4/4 | 2.925 s |
| MolmoAct2 | 1/1 | 1/1 | 1/1 | 1/1 | 4/4 | 5.196 s |

这只是端到端冒烟结果。每个比例的分母只有 1，统计方差极大，不能得出 VLA-JEPA 或
MolmoAct2 质量一定高于 Pi0.5 的结论。X-VLA 在相同 task/seed 上用官方
`lerobot-eval` 交叉检查也为 0/1，说明失败不是自定义适配器独有，但不代表正式成功率为
0%。

## 8. 证据强度与限制

与跨任务辅助模型混排相比，这组实验更有说服力：五个模型在相同四 suite、相同环境状态
和相同动作语义下闭环执行，并统一了推理应用边界。不过仍有以下限制：

1. 原生 action chunk 为 7、10、30 或 50 步，X-VLA 还使用 20 维跨 embodiment 表示；
2. 模型精度不同：Pi0.5 为 FlashRT FP8，X-VLA 主要为 FP32，其余以 BF16 为主；
3. 推理/flow 步数为 4、8 或 10；
4. 当前闭环 quick 每模型只有四个 episode，尚未完成正式 400 episodes 和多 seed。

因此性能部分可以比较现成部署路径的延迟、显存和 kernel 行为；机器人策略质量需要以
`full` 闭环结果为准。最终选型还应增加 P99、功耗和每个成功 episode 的能耗。

## 9. 优化建议

1. Pi0.5：保留当前 FlashRT FP8 路径，并用真实 LIBERO 观测做闭环精度回归。
2. SmolVLA：优先尝试 CUDA Graph、`torch.compile`、固定 shape 和减少 flow steps，重点
   降低 13,646 次 kernel 发射与调度间隙。
3. X-VLA：先完成 400-episode 质量复验，再验证 BF16/FP8 部署和主要 GEMM/attention
   融合。
4. VLA-JEPA：重点压缩 CPU/框架到 GPU 的调度间隙，分析小 grid 卷积和 world-model
   分支能否图捕获。
5. MolmoAct2：优先减少八次 flow 中的 35,080 次 kernel 发射，并优化 BF16
   attention/GEMM。

## 10. 复现与归档

```bash
cd /home/zhangyunli/FlashRT

# 五模型性能
scripts/robot_bench_latency.sh all

# 五模型 NSys + NCU
scripts/robot_bench_nsight.sh all

# 20-episode 冒烟或 2000-episode 正式闭环
scripts/robot_bench_e2e.sh quick
scripts/robot_bench_e2e.sh full
```

容器进入方式、固定 revision、每个 `.rep` 的测试用例和完整代码改动清单见
`docs/robot_model_experiment_reproduction.md`。

```text
/home/zhangyunli/FlashRT-profile-artifacts-20260804/robot_profile/
├── libero_aligned/       # latency、NSys、NCU
└── libero_e2e_quick/     # 当前闭环 quick JSON
```

官方协议参考：<https://huggingface.co/docs/lerobot/libero>、
<https://huggingface.co/docs/lerobot/env_processor>。
