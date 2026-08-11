# 五机器人模型 Nsight 性能分析报告

分析日期：2026-08-11
GPU：NVIDIA GeForce RTX 5090（SM120，32 GB）
驱动：590.44.01
Nsight Systems：2025.5.1.121
Nsight Compute：2025.3.1.0

## 1. 分析范围

五个模型各从正式六组实验中选择同一组 `libero_spatial:0`，使用 seed 0 的真实初始
双相机画面、真实机器人状态和完整任务文本：

> pick up the black bowl between the plate and the ramekin and place it on the plate

选择同一组而不是给每个模型挑不同动作，是为了让图像尺寸、文本长度和环境状态尽量
一致。闭环部分分析该动作的 10 个 episodes；Nsight 部分捕获该真实初始观测的一次
完整 action-chunk 重规划。没有把整个 episode 放进 NCU replay，因为环境 step 和数百
次重规划会造成不可控的报告体积，也会因 replay 改变仿真时序。

## 2. 给非专业读者的阅读指南

一次机器人策略推理可粗略理解为：

```text
相机/状态/文字
      ↓ CPU 预处理与分词
      ↓ 把数据送到 GPU
      ↓ GPU 执行许多 kernel
      ↓ 同步并取回 action chunk
      ↓ 环境连续执行若干步，再重新规划
```

- **Kernel**：GPU 上一次具体计算任务。矩阵乘法、注意力、归一化、复制通常是不同
  kernel。一次模型推理可能发射几千到几万个 kernel。
- **CUDA launch**：CPU 告诉 GPU“执行这个 kernel”。单次开销不大，但小 kernel 太多
  时，通知成本会累积成瓶颈。
- **CUDA Graph**：提前记录一串固定 GPU 工作，后续一次提交整张图，能减少大量 launch。
- **同步**：CPU 等 GPU 完成。同步时间长不一定是同步函数本身慢，而是 CPU 在这里等待
  前面的 GPU 工作。
- **GEMM/MMA**：神经网络最常见的矩阵乘法；Tensor Core 擅长执行 MMA。
- **Attention**：视觉/语言 token 之间建立关联的计算，长序列时成本会上升。
- **Occupancy（占用率）**：GPU 同时驻留了多少可执行线程束。低占用率可能来自寄存器
  太多、网格太小或共享内存限制，但低占用率本身不一定等于低性能。
- **P50/P95**：典型延迟与尾延迟。机器人控制既需要平均快，也需要少卡顿。

Nsight Systems（NSys）用于观察完整 CPU/GPU 时间线和全部 kernel；Nsight Compute
（NCU）对一枚代表性热点 kernel 做多次 replay，读取硬件计数器。NCU 的单 kernel
结论不会被误写成“整个模型的 GPU 利用率”。

## 3. 测量协议

| 项目 | 设置 |
|---|---|
| 输入 | `libero_spatial:0` seed 0 真实初始观测 |
| 相机 | agent view + wrist view，256×256 |
| 输出对齐 | 完整原生 action chunk，统一检查第一步 7 维动作 |
| 稳态延迟 | warmup 10，测量 30 次 |
| NSys | warmup 3，捕获 1 次完整 replan |
| NSys trace | CUDA、NVTX、OS runtime、cuBLAS、cuDNN |
| NCU | kernel replay，full metric set，每模型一枚代表性热点 kernel |
| GPU 条件 | 正式采集前确认无其他计算进程 |

推理计时包含图像 resize、分词、模型预处理和 CPU/GPU 搬运，不包含 MuJoCo environment
step。`kernel/P50` 是独立 NSys 与延迟运行组合出的近似占比，只用于判断瓶颈方向；CPU
工作与 GPU 工作可能重叠，因此“非 kernel 时间”不是可以逐项精确相加的账本。

## 4. 总览

<!-- FINAL_PERFORMANCE_OVERVIEW -->

<!-- FINAL_NSYS_OVERVIEW -->

## 5. Pi0.5：代表实验详细分析

<!-- FINAL_PI05_ANALYSIS -->

## 6. SmolVLA：代表实验详细分析

<!-- FINAL_SMOLVLA_ANALYSIS -->

## 7. GR00T N1.7：代表实验详细分析

<!-- FINAL_GROOT_ANALYSIS -->

## 8. VLA-JEPA：代表实验详细分析

<!-- FINAL_VLA_JEPA_ANALYSIS -->

## 9. MolmoAct2：代表实验详细分析

<!-- FINAL_MOLMOACT2_ANALYSIS -->

## 10. 横向瓶颈与优化优先级

<!-- FINAL_CROSS_MODEL_ANALYSIS -->

## 11. 分析限制

1. NSys 只捕获一次 replan，延迟表用 30 次稳态分布补足单样本不足。
2. NCU 只 replay 一枚代表 kernel；它用于解释该热点的硬件行为，不能替代 NSys 的
   全模型时间占比。
3. 模型 action chunk 长度和去噪步数不同。这里比较的是模型按原生配置生成一个完整
   chunk 的部署延迟，不是等 token、等 FLOP 的微基准。
4. `libero_spatial:0` 的真实文本比旧辅助 profile 的短提示更长，本报告只使用新的真实
   输入采集，不把旧随机图像数据混入正式表格。
5. 性能优化建议必须经过成功率回归；减少去噪步数、改变精度或图捕获边界都可能改变
   动作质量。

## 12. 原始报告

```text
/workspace/models/robot-bench-results/final_20260811/nsight/<model>/
```

每个模型目录含 `.nsys-rep`、`.ncu-rep`、SQLite、CSV 和运行 JSON。复现命令与宿主机
归档路径见同目录 `README.md`。
