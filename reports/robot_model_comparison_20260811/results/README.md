# 实验结果归档说明

本目录是提交时从容器
`/workspace/models/robot-bench-results/` 完整复制的实验结果快照。目录中不包含模型权重、
LIBERO 数据集、Hugging Face 缓存或 Python 缓存。

## 当前状态

- `final_20260811/e2e/quick/pi05.json` 使用最终约定：每个任务使用环境 seed 0–9，策略
  随机流只在模型进程启动时固定一次。该结果为 21/60，成功率 35%。
- `final_20260811/e2e/quick/groot_libero.json` 和 `smolvla_libero.json` 来自较早的“每个
  episode 同时重置环境和策略随机流”协议，不能与上述 Pi0.5 结果直接作为最终排名。
- SmolVLA 按最终协议的重跑在中途被暂停，没有产生可提交的完整 JSON。
- GR00T、SmolVLA、VLA-JEPA 和 MolmoAct2 仍需按最终协议重跑；五模型统一的最新
  latency、NSys 和 NCU 结果也尚未采集完成。
- `groot_validation_20260811/` 是 GR00T 检查点接入后的单任务可运行性验证。

## 历史目录

- `e2e*`、`latency*`、`nsys_20260810/` 和 `nsight/` 是协议整理过程中的历史结果。
- `*_replacement_20260811/` 是使用 GR00T 替代 X-VLA 后的初步性能采样。
- X-VLA 结果仅保留为排除该模型的审计依据，不属于正式五模型排名。

## 文件用途

- `*.json`：实验配置、逐 episode 结果和延迟统计。
- `*.csv`：从 Nsight 导出的 CUDA API、GPU kernel 和 NCU 指标，便于直接复核报告。
- `*.nsys-rep`：可用 Nsight Systems GUI 打开的 CPU/GPU 时间线。
- `*.ncu-rep`：可用 Nsight Compute GUI 打开的代表性 kernel 分析。
- `*.sqlite`：NSys 报告导出的查询数据库。

恢复实验后，统一复现命令见上一级 `README.md`。新结果应写入
`/workspace/models/robot-bench-results/final_20260811/`，确认五个模型均完成后再更新本
归档和两份正式报告。
