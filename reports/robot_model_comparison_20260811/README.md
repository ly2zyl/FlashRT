# 五机器人模型实验复现说明与改动清单

更新日期：2026-08-11

## 报告索引

- `robot_model_test_report.md`：六个代表动作的正确性、成功率和性能对比。
- `robot_model_performance_analysis.md`：五个模型各取一组真实实验输入的
  Nsight Systems / Nsight Compute 详细分析。
- `results/`：从容器归档的端到端、延迟、NSys 和 NCU 原始结果及导出表格。
- 本文件：容器进入方法、一键复现命令、固定版本、结果位置和代码改动清单。

正式报告只放在本目录，不再放入 `docs/`。

## 1. 正式对比组

| 模型 | 脚本标识 | 本地检查点 |
|---|---|---|
| Pi0.5 | `pi05` | `/workspace/models/pi05_libero_finetuned_v044` |
| SmolVLA | `smolvla_libero` | `/workspace/models/SmolVLA-LIBERO` |
| GR00T N1.7 | `groot_libero` | `/workspace/models/GR00T-N1.7-LIBERO/<suite>` |
| VLA-JEPA | `vla_jepa_libero` | `/workspace/models/VLA-JEPA-LIBERO` |
| MolmoAct2 | `molmoact2_libero` | `/workspace/models/MolmoAct2-LIBERO-LeRobot` |

X-VLA 不再属于正式对比组：当前检查点在两轮快速测试、60-episode 代表性测试以及
官方 `lerobot-eval` 单例复核中均未成功。它可以快速产生有限数值动作，但在当前
checkpoint、动作转换和 LIBERO 环境组合下没有形成有效闭环能力，因而继续比较它的
推理速度没有实际意义。

GR00T 使用 NVIDIA 发布的四个 suite 专用检查点，脚本按测试用例自动选择 Spatial、
Object、Goal 或 LIBERO-10 权重，最终仍按一个模型家族统计。需要注意，这能证明当前
可用模型家族的端到端能力，但由于各模型的训练数据和训练流程并不相同，成功率差异
不能完全归因于网络结构。

## 2. 容器与进入方法

所有实验统一使用：

```text
容器：      flashrt-robot-bench
镜像：      flashrt:5090
GPU：       NVIDIA GeForce RTX 5090，SM120，32 GB
源码目录：  /workspace/FlashRT
模型目录：  /workspace/models
模型卷：    flashrt-pi05-models
```

在宿主机执行：

```bash
cd /home/zhangyunli/FlashRT

# 查看容器状态
scripts/robot_bench_container.sh status

# 创建或启动容器、补齐依赖和缺失权重，然后进入容器
scripts/robot_bench_container.sh enter
```

也可以不手动进入容器，直接运行后文三个实验脚本；脚本会自动启动相同容器。第一次
运行需要下载模型，GR00T 四套 suite 权重体积较大；下载支持断点续传，脚本会校验
固定 revision 和 `model.safetensors` 的 SHA-256。

## 3. 一键复现实验

### 3.1 正确性与闭环性能测试

```bash
cd /home/zhangyunli/FlashRT

# 五模型 × 六个动作 × 每动作 10 episodes，共 300 episodes
scripts/robot_bench_e2e.sh quick \
  /workspace/models/robot-bench-results/final_20260811/e2e
```

`quick` 固定测试：

```text
libero_spatial:0
libero_object:0
libero_goal:0
libero_goal:7
libero_10:0
libero_10:9
```

每个动作的环境使用 seed 0–9。每个模型进程启动时固定一次 PyTorch/CUDA 策略随机流，
随后按固定用例顺序连续推进；这既能重复整次运行，也符合长生命周期策略部署。输入为
256×256 双相机画面、机器人状态和语言指令，环境接收统一的 7 维相对动作。JSON 同时
记录成功判定、步数、墙钟、控制频率、完成步数和观测到动作耗时。

如需运行 LeRobot 推荐的完整四套件协议：

```bash
# 五模型 × 40 tasks × 每 task 10 episodes，共 2000 episodes
scripts/robot_bench_e2e.sh full \
  /workspace/models/robot-bench-results/full_20260811/e2e
```

完整协议耗时很长。本报告使用六动作代表集做候选模型筛查，不把它等同于完整 400
episodes/模型的标准 LIBERO 结果。

### 3.2 稳态推理性能

```bash
scripts/robot_bench_latency.sh all \
  /workspace/models/robot-bench-results/final_20260811/latency
```

五个模型都读取 `libero_spatial:0`、seed 0 的真实初始双相机画面、真实任务文本和
机器人状态。计时边界包含图像 resize、tokenization、CPU/GPU 搬运和完整 action-chunk
重规划，不包含 MuJoCo environment step。每个模型 warmup 10 次，再正式测量 30 次，
输出 P50/P90/P95、重规划频率、冷加载、首次推理和峰值显存。

### 3.3 Nsight Systems 与 Nsight Compute

```bash
# 五个模型依次采集 NSys；并对每个模型的一枚代表性热点 kernel 做 NCU full 分析
scripts/robot_bench_nsight.sh all \
  /workspace/models/robot-bench-results/final_20260811/nsight full

# 如果只需要整条 CPU/GPU 时间线，不运行耗时更长的 NCU
scripts/robot_bench_nsight.sh all \
  /workspace/models/robot-bench-results/final_20260811/nsight_systems systems
```

Nsight 同样使用六组正式实验中的 `libero_spatial:0` seed 0 真实初始观测。NSys 捕获
一次完整重规划，可回答“时间花在哪里”；NCU 对一枚热点 kernel 做 replay，可回答
“这枚 kernel 为什么没有跑满”。NCU 单 kernel 指标不能代表整个模型，必须结合 NSys
的全路径占比解释。

## 4. 默认结果位置

```text
/workspace/models/robot-bench-results/final_20260811/
├── e2e/quick/<model>.json
├── latency/<model>/latency.json
└── nsight/<model>/
    ├── nsys.nsys-rep
    ├── nsys.sqlite
    ├── nsys_run.json
    ├── nsys_stats_cuda_gpu_kern_sum.csv
    ├── nsys_stats_cuda_api_sum.csv
    ├── ncu.ncu-rep
    ├── ncu_run.json
    └── ncu_details.csv
```

源码树外的宿主机归档位置：

```text
/home/zhangyunli/FlashRT-profile-artifacts-20260804/
└── robot_profile/final_20260811/
```

`.nsys-rep` 可用 Nsight Systems GUI 打开，`.ncu-rep` 可用 Nsight Compute GUI 打开；
CSV 和 JSON 用于报告表格复核。

提交时的结果快照同时归档在本目录的 `results/` 下，目录结构保持与容器一致。归档状态
和不同批次的协议差异见 `results/README.md`。其中 `final_20260811` 是实验暂停时的快照，
尚不能作为五模型最终横向结论；恢复测试后应按第 3 节命令覆盖生成完整结果。

## 5. 固定模型版本

| 模型/组件 | Revision |
|---|---|
| SmolVLA LIBERO | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| GR00T N1.7 base | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| GR00T Spatial | `32a6ec786d6509df31b40392b4e4dcdda78c0f11` |
| GR00T Object | `1499db357f6ca3762b56c2e8c00b530eb9a09444` |
| GR00T Goal | `436c57c0eb7a90be54270abf3977668b2084ad75` |
| GR00T LIBERO-10 | `5ee08ab09fac5c5ef2388a14c882ea825ac861db` |
| VLA-JEPA LIBERO | `735d9f692981e286ade093b5046627eda876e5d0` |
| MolmoAct2 LeRobot | `f0c5a9567c2b72faadec901e16e055c8b098c2f5` |
| LIBERO assets | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |

Pi0.5 使用仓库已有的本地 `pi05_libero_finetuned_v044`，不改变其原有源码、配置或入口。
VLA-JEPA 内部依赖的 Qwen3-VL-2B 和 V-JEPA2 encoder 是策略组件，不是第六、第七个
对比模型。GR00T 的 processor 复用相同固定版本的 Qwen3-VL tokenizer/image-processor
资产，但不加载该 Qwen 模型执行独立推理。

## 6. 本次新增或整理的代码

本次没有修改 FlashRT 原有模型实现、原有 Pi0/Pi0.5 配置、入口或测试。正式实验新增：

| 文件 | 职责 |
|---|---|
| `benchmarks/libero_e2e_eval.py` | 五模型真实 LIBERO 闭环评测、固定 seed、成功率和闭环性能记录 |
| `benchmarks/libero_policy_profile.py` | 五模型统一真实输入边界的延迟、显存和 Nsight workload |
| `scripts/robot_bench_container.sh` | 唯一容器入口；启动、进入、依赖准备、固定权重下载与哈希校验 |
| `scripts/robot_bench_e2e.sh` | 正确性/闭环性能脚本；提供 `quick` 与 `full` 两种协议 |
| `scripts/robot_bench_latency.sh` | 稳态延迟、重规划频率和峰值显存脚本 |
| `scripts/robot_bench_nsight.sh` | NSys/NCU 性能分析脚本 |
| `reports/robot_model_comparison_20260811/` | 最终报告、复现说明和改动清单 |

应用户要求，本次提交还包含工作区已有的 `docker/Dockerfile` 和 `AGENTS.md`，便于在
同一实验分支保存当前项目状态；二者不是机器人评测脚本本身的一部分。

## 7. 清理边界

- Pi0 只删除下载权重和本实验痕迹；原有源码、配置、入口和测试全部保留。
- Pi0.5 权重保留并进入正式对比。
- 独立 Qwen 对比模型及其权重已移除；VLA-JEPA 必需的内部 Qwen 组件保留。
- X-VLA 已从正式脚本和模型枚举移除；历史 0/60 JSON 只作为“为何排除”的审计依据，
  不进入正式排名。
- 不保留 PushT、ACT、Diffusion Policy、VQ-BeT 或其他辅助模型组。

## 8. 官方协议参考

- [LeRobot LIBERO 文档](https://huggingface.co/docs/lerobot/libero)：标准 suite、双相机、
  8 维状态、7 维动作和推荐评测规模。
- [LeRobot GR00T 文档](https://huggingface.co/docs/lerobot/groot)：GR00T N1.7 集成、
  官方 suite 专用检查点与初步 LIBERO 成绩。
