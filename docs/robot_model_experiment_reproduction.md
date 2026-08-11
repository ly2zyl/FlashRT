# 五模型 LIBERO 实验复现与改动清单

更新日期：2026-08-11

## 1. 最终实验范围

只保留五个能够在同一 LIBERO 闭环环境中端到端运行的机器人策略：

| 模型 | 本地检查点 | 模型标识 |
|---|---|---|
| Pi0.5 | `/workspace/models/pi05_libero_finetuned_v044` | `pi05` |
| SmolVLA | `/workspace/models/SmolVLA-LIBERO` | `smolvla_libero` |
| GR00T N1.7 | `/workspace/models/GR00T-N1.7-LIBERO/<suite>` | `groot_libero` |
| VLA-JEPA | `/workspace/models/VLA-JEPA-LIBERO` | `vla_jepa_libero` |
| MolmoAct2 | `/workspace/models/MolmoAct2-LIBERO-LeRobot` | `molmoact2_libero` |

四组端到端任务是 LIBERO 的四个标准 suite：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

五个模型共享相同环境、任务、种子、双相机观测、8 维机器人状态和相对控制模式，
最终都向环境提交 7 维动作。模型内部的 action chunk 长度、精度和推理步数仍不同，
因此延迟比较表示“现成策略完成一次 replan 的部署成本”，不是等 FLOP 的算子对比。

GR00T 使用 NVIDIA 官方发布的四个 suite 专用检查点，评测脚本按当前 suite 自动路由，
统计时仍作为一个模型。X-VLA 因连续 60 个代表性 episode 全部失败，且官方
`lerobot-eval` 同任务复核仍失败，已退出正式对比组。

不再保留 GR00T N1.6、ACT、Diffusion Policy、VQ-BeT 或 PushT 辅助组。独立 Qwen
模型不输出机器人动作，已从本实验剔除。VLA-JEPA 自身需要的 Qwen3-VL-2B backbone
是策略组件，不是独立对比模型，必须保留。

## 2. 容器与进入方式

所有实验使用同一个容器：

```text
container: flashrt-robot-bench
image:     flashrt:5090
GPU:       --gpus all
NCU:       --cap-add SYS_ADMIN
workspace: /workspace/FlashRT
models:    /workspace/models
```

在宿主机执行：

```bash
cd /home/zhangyunli/FlashRT

# 查看状态
scripts/robot_bench_container.sh status

# 自动创建/启动容器、准备依赖和缺失权重，然后进入 bash
scripts/robot_bench_container.sh enter
```

也可以不进入容器，直接运行下面三个实验脚本；它们会自动启动容器。

源码通过 bind mount 映射到 `/workspace/FlashRT`。模型和容器内结果保存在 Docker volume
`flashrt-pi05-models`。FlashRT 编译扩展从仓库外的归档目录只读挂载，实验不会向
`flash_rt/` 写入 `.so`。

入口脚本第一次运行时会准备 `/opt/lerobot-venv`、无头 EGL、LIBERO 环境和固定 revision
的模型依赖。LIBERO 仿真资源固定为 `lerobot/libero-assets` revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。

## 3. 一键复现实验

### 3.1 闭环端到端结果

代表性 quick：五模型 × 六个代表动作 × 每个动作 10 episodes，共 300 episodes。

```bash
cd /home/zhangyunli/FlashRT
scripts/robot_bench_e2e.sh quick
```

正式测试：五模型 × 四个 suite × 每 suite 10 个 task × 每 task 10 episodes，共
2000 episodes，即每个模型 400 episodes。

```bash
scripts/robot_bench_e2e.sh full
```

六个动作覆盖空间抓放、物体抓放、开抽屉、开炉灶、双物体搬运和微波炉复合动作。
`full` 需要数小时；代表性 quick 适合先筛选候选模型，但仍不能替代完整 40-task 协议。

### 3.2 推理性能

延迟、吞吐和峰值显存与闭环质量分开测量：

```bash
# 一次测试全部五个模型
scripts/robot_bench_latency.sh all

# 或只测一个模型
scripts/robot_bench_latency.sh pi05
scripts/robot_bench_latency.sh smolvla_libero
scripts/robot_bench_latency.sh groot_libero
scripts/robot_bench_latency.sh vla_jepa_libero
scripts/robot_bench_latency.sh molmoact2_libero
```

统一性能协议为 seed 0、warmup 10 次、正式测量 30 次。输入是同一对内存中的
`float32 [1,3,224,224]` CPU 图像和同一文本指令，计时包含预处理、tokenization、
CPU/GPU 搬运以及完整 replan，输出统一截取第一步 7 维动作。磁盘图像解码和环境 step
不在性能计时区。

### 3.3 Nsight Systems 与 Nsight Compute

```bash
# 五模型依次生成 NSys 和 NCU 报告
scripts/robot_bench_nsight.sh all

# 或只分析一个模型
scripts/robot_bench_nsight.sh pi05
scripts/robot_bench_nsight.sh vla_jepa_libero
```

NSys 捕获一次完整 replan。NCU 使用 full metric set 和 kernel replay，针对每个模型的
一个代表性 GPU kernel；NCU 会明显慢于普通延迟测试。

## 4. 模型固定版本和输入输出

| 模型 | 固定版本/依赖 | 原生策略输出 | 环境输出 |
|---|---|---|---|
| Pi0.5 | 本地 `pi05_libero_finetuned_v044` | `10×7`，10 次去噪，FlashRT FP8 | 每步 `7` 维 |
| SmolVLA | `lerobot/smolvla_libero@31d453f7` | `1×50×7`，10 次 flow | 每步 `7` 维 |
| GR00T N1.7 | 四个 `nvidia/gr00t17-lerobot-libero_*@固定 revision` | `1×16×7`，4 次 flow | 每步 `7` 维 |
| VLA-JEPA | `lerobot/VLA-JEPA-LIBERO@735d9f69` | `1×7×7`，4 次推理 | 每步 `7` 维 |
| MolmoAct2 | `allenai/MolmoAct2-LIBERO-LeRobot@f0c5a956` | `1×10×7`，8 次 flow | 每步 `7` 维 |

GR00T 的四个官方检查点固定为：

| Suite | Hugging Face 仓库 | Revision |
|---|---|---|
| Spatial | `nvidia/gr00t17-lerobot-libero_spatial-640` | `32a6ec786d6509df31b40392b4e4dcdda78c0f11` |
| Object | `nvidia/gr00t17-lerobot-libero_object-640` | `1499db357f6ca3762b56c2e8c00b530eb9a09444` |
| Goal | `nvidia/gr00t17-lerobot-libero_goal-640` | `436c57c0eb7a90be54270abf3977668b2084ad75` |
| LIBERO-10 | `nvidia/gr00t17-lerobot-libero_10-640` | `5ee08ab09fac5c5ef2388a14c882ea825ac861db` |

离线运行所需依赖也固定在模型 volume 中：

- SmolVLA：`SmolVLM2-500M-Video-Instruct@7b375e1b`；
- GR00T N1.7：`GR00T-N1.7-3B@2fc962b9` base；输入编码复用容器内固定版本的
  Qwen3-VL processor 资产，不加载其模型权重。该固定 revision 的 `tokenizer.json`、
  `vocab.json`、`merges.txt`、`chat_template.json` 和 `preprocessor_config.json` 与
  Cosmos-Reason2-2B 对应文件的 Hub blob ID 完全相同；
- VLA-JEPA：Qwen3-VL-2B `@89644892` 和 V-JEPA2 encoder `@b3c1679b`；
- MolmoAct2：LIBERO base `@0d24a92b` 和 FAST tokenizer `@d45593b4`。

## 5. 当前已完成结果

### 5.1 端到端快速测试

原五模型组已于 2026-08-10 完成 300 个闭环 episodes。X-VLA 已在 2026-08-11
退出正式对比组；GR00T N1.7 的替换结果以最新代表性报告为准：

| 模型 | 成功/episode | 代表集成功率 | 聚合控制频率 |
|---|---:|---:|---:|
| Pi0.5 | 18/60 | 30% | 63.44 Hz |
| SmolVLA | 42/60 | 70% | 53.21 Hz |
| GR00T N1.7 | 见最新报告 | 见最新报告 | 见最新报告 |
| VLA-JEPA | 60/60 | 100% | 30.86 Hz |
| MolmoAct2 | 60/60 | 100% | 21.14 Hz |

逐动作成功率、完成步数、控制频率和模型分析见
`docs/robot_model_representative_quick_report_20260810.md`；性能与 NSys 分析见
`docs/robot_model_quick_nsys_report_20260810.md`。

X-VLA 的 60 个 episode 都运行到最大步数，同一 task/seed 使用官方 `lerobot-eval`
复核也是 0/1。它在当前检查点、动作转换和环境组合下不能形成有效对比，因此不再用
更多算力扩大该失败样本。

### 5.2 当前轮性能和 Nsight（共享 GPU 条件）

| 模型 | P50 / P95 | 峰值 allocated | NSys kernel 总时间 / 实例数 |
|---|---:|---:|---:|
| Pi0.5 | 21.86 / 22.46 ms | 9.03 GiB | 20.215 ms / 2,735 |
| SmolVLA | 134.36 / 144.36 ms | 0.90 GiB | 28.353 ms / 13,646 |
| GR00T N1.7 | 见最新报告 | 见最新报告 | 见最新报告 |
| VLA-JEPA | 63.61 / 66.53 ms | 5.86 GiB | 20.256 ms / 4,588 |
| MolmoAct2 | 153.31 / 284.62 ms | 11.10 GiB | 134.548 ms / 35,080 |

本轮 GPU 有其他计算进程，绝对值需在独占 GPU 后复验。当前全面热点分析见
`docs/robot_model_quick_nsys_report_20260810.md`；2026-08-04 独占基线见
`docs/robot_model_performance_report_20260804.md`。

## 6. 输出位置

容器内默认结果：

```text
/workspace/models/robot-bench-results/
├── e2e/
│   ├── quick/<model>.json
│   └── full/<model>.json
├── latency/<model>/latency.json
└── nsight/<model>/
    ├── nsys.nsys-rep
    ├── nsys_stats_cuda_gpu_kern_sum.csv
    ├── nsys_stats_cuda_api_sum.csv
    ├── ncu.ncu-rep
    └── ncu_details.csv
```

当前正式归档在源码树外：

```text
/home/zhangyunli/FlashRT-profile-artifacts-20260804/robot_profile/
├── libero_aligned/       # 旧五模型 latency、NSys、NCU 历史归档
├── libero_e2e_quick/     # 前次五模型闭环 quick JSON
├── libero_quick_nsys_20260810/  # 前次四用例 quick、频率、NSys 和报告
└── libero_representative_quick_20260810/  # 当前六用例 × 10 episodes
```

## 7. 本次代码文件清单

本次机器人对比只新增以下文件，没有修改 FlashRT 原有模型实现、配置、入口或原有测试：

| 文件 | 职责 |
|---|---|
| `benchmarks/libero_policy_profile.py` | 五模型统一应用边界的性能/Nsight 测试用例 |
| `benchmarks/libero_e2e_eval.py` | 五模型真实 LIBERO MuJoCo 闭环评测 |
| `scripts/robot_bench_container.sh` | 唯一容器入口；创建、启动、进入、依赖和权重准备 |
| `scripts/robot_bench_latency.sh` | 五模型延迟、吞吐、显存测试 |
| `scripts/robot_bench_nsight.sh` | 五模型 NSys/NCU 性能分析 |
| `scripts/robot_bench_e2e.sh` | 五模型 quick/full 闭环端到端测试 |
| `docs/robot_model_experiment_reproduction.md` | 本复现说明和改动清单 |
| `docs/robot_model_performance_report_20260804.md` | 性能与 Nsight 测试报告 |
| `docs/robot_model_quick_nsys_report_20260810.md` | 前次 quick、频率和全面 NSys 对比报告 |
| `docs/robot_model_representative_quick_report_20260810.md` | 六类代表动作 × 10 episodes 结果分析 |

工作区原有的 `docker/Dockerfile` 和 `AGENTS.md` 不属于本次实验改动，未触碰。

## 8. 已清理内容和保留边界

已从 Docker volume 删除下列辅助模型权重、缓存和对应结果，合计约 7.6 GB；需要时只能
重新下载：

```text
/workspace/models/GR00T-N1.6-3B
/workspace/models/qwen3-1.7b-tokenizer
/workspace/models/ACT-Aloha-TransferCube
/workspace/models/Diffusion-PushT
/workspace/models/VQBeT-PushT
/workspace/models/torch-cache/hub/checkpoints/resnet18-f37072fd.pth
```

同时删除本次曾新增但已无用途的辅助 benchmark：

```text
benchmarks/robot_model_profile.py
benchmarks/act_profile.py
benchmarks/lerobot_pusht_profile.py
```

其中 `benchmarks/robot_model_profile.py` 指的是本轮新增的辅助文件；仓库若原本存在同名
受版本控制文件则不应删除。当前工作树中不存在该文件。

Pi0 只删除了下载权重和缓存；Pi0、Pi0-fast、Pi0.5 原有源码、配置、入口和测试全部
保留。Pi0.5 权重仍在 `/workspace/models/pi05_libero_finetuned_v044`。

独立 Qwen 对比权重仍保持删除。唯一保留的 Qwen 权重位于
`/workspace/models/VLA-JEPA-deps/Qwen3-VL-2B-Instruct`，仅供 VLA-JEPA 策略内部使用。

## 9. 官方协议参考

LeRobot 的 LIBERO 文档定义了四个标准 suite、双相机观测、8 维状态、7 维动作，以及
每个模型 400 episodes 的标准评测规模：

- <https://huggingface.co/docs/lerobot/libero>
- <https://huggingface.co/docs/lerobot/env_processor>
- <https://huggingface.co/docs/lerobot/groot>
