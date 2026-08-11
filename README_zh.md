# FlashRT 中文部署与机器人模型评测说明

本文说明如何在 NVIDIA GPU 服务器上部署 FlashRT，以及如何复现本分支新增的五个机器人
模型 LIBERO 端到端与性能分析实验。上游项目的完整功能、模型列表和 API 仍以
[`README.md`](README.md) 为准。

## 1. 本分支包含什么

机器人模型正式对比组为：

| 模型 | 脚本标识 | 用途 |
|---|---|---|
| Pi0.5 | `pi05` | FlashRT 优化路径与主要对比对象 |
| SmolVLA | `smolvla_libero` | 轻量 VLA 基线 |
| GR00T N1.7 | `groot_libero` | NVIDIA VLA，每个 LIBERO suite 使用专用权重 |
| VLA-JEPA | `vla_jepa_libero` | 视觉表征与动作策略基线 |
| MolmoAct2 | `molmoact2_libero` | 大型视觉语言动作模型基线 |

本分支新增了三条相互独立的实验链路：

- `scripts/robot_bench_e2e.sh`：LIBERO 闭环正确性和控制性能测试。
- `scripts/robot_bench_latency.sh`：真实观测输入下的稳态推理延迟、频率和显存测试。
- `scripts/robot_bench_nsight.sh`：Nsight Systems 与 Nsight Compute 性能分析。

已有报告和原始结果位于
[`reports/robot_model_comparison_20260811/`](reports/robot_model_comparison_20260811/)。其中
`results/` 是暂停实验时的快照，不代表五个模型均已按最终协议跑完；具体状态见
[`results/README.md`](reports/robot_model_comparison_20260811/results/README.md)。

## 2. 推荐部署环境

本实验实际使用以下环境：

| 项目 | 配置 |
|---|---|
| 操作系统 | x86_64 Linux |
| GPU | NVIDIA GeForce RTX 5090，32 GB，计算能力 SM120 |
| 驱动 | 590.44.01 |
| 容器基础环境 | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Python | 容器内 Python 3.12 |
| Docker 镜像 | `flashrt:5090` |
| 实验容器 | `flashrt-robot-bench` |
| 模型卷 | `flashrt-pi05-models` |

在其他 NVIDIA GPU 上可以构建 FlashRT，但本分支的五模型对比脚本和结果只在 RTX 5090
上验证过。完整模型、依赖和缓存目前约占 100 GiB，建议至少预留 120 GiB 可用磁盘。
MolmoAct2 等模型也需要较大显存；低于 32 GB 时可能需要逐模型调整加载方式，不能保证
直接运行五模型脚本。

宿主机需要：

1. 可用的 NVIDIA 驱动，`nvidia-smi` 能正确识别 GPU。
2. Docker Engine。
3. NVIDIA Container Toolkit，使 `docker run --gpus all` 可用。
4. 能访问 NVIDIA NGC、Hugging Face 和 Google Storage 的网络。
5. 本地 Pi0.5 微调检查点；该权重不会由脚本自动下载。

先验证 Docker GPU：

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

如果这一步失败，应先修复 NVIDIA Container Toolkit，不要继续构建实验环境。

## 3. 获取实验分支

```bash
git clone --branch agent/robot-model-benchmark \
  git@github.com:ly2zyl/FlashRT.git
cd FlashRT

git branch --show-current
git log -1 --oneline
```

预期分支为 `agent/robot-model-benchmark`。本文档对应的首个完整实验提交为
`112119503a64e4f458428cd76d5c5c89e1e4ea10`，后续 README 修订会产生更新的提交。

## 4. 构建 RTX 5090 Docker 镜像

在仓库根目录执行：

```bash
docker build -t flashrt:5090 \
  --build-arg GPU_ARCH=120 \
  --build-arg BUILD_JOBS=8 \
  -f docker/Dockerfile .
```

`GPU_ARCH=120` 固定生成 RTX 5090 所需的 SM120 代码。`BUILD_JOBS` 用于限制并行编译
进程，防止宿主机内存不足；内存较少时可改为 `4` 或 `2`。首次构建需要拉取 NGC 镜像
并编译 CUDA 扩展，耗时取决于网络、CPU 和磁盘性能。

验证镜像和 CUDA 扩展：

```bash
docker run --rm --gpus all -i flashrt:5090 python3 - <<'PY'
import torch
import flash_rt
from flash_rt import flash_rt_fa2, flash_rt_kernels

print("FlashRT:", flash_rt.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA 可用:", torch.cuda.is_available())
print("kernels:", flash_rt_kernels.__file__)
print("fa2:", flash_rt_fa2.__file__)
PY
```

### 导出机器人容器使用的扩展

实验容器会把当前源码以 bind mount 方式挂载进去，因此需要另外挂载与镜像匹配的二进制
扩展。默认位置是仓库同级的 `FlashRT-profile-artifacts-20260804/build/`：

```bash
export ROBOT_BENCH_ARTIFACTS="$(pwd)-profile-artifacts-20260804"
mkdir -p "${ROBOT_BENCH_ARTIFACTS}/build"

docker run --rm \
  --mount "type=bind,src=${ROBOT_BENCH_ARTIFACTS}/build,dst=/out" \
  flashrt:5090 bash -lc \
  'cp /workspace/FlashRT/flash_rt/flash_rt_kernels*.so \
      /workspace/FlashRT/flash_rt/flash_rt_fa2*.so /out/'

ls -lh "${ROBOT_BENCH_ARTIFACTS}/build"
```

目录中应各有一个 `flash_rt_kernels*.so` 和 `flash_rt_fa2*.so`。重新构建镜像后，应重新
导出扩展，避免源码、Python ABI 和二进制不一致。

## 5. 准备模型和 LIBERO 环境

### 5.1 放入 Pi0.5 本地权重

脚本使用固定的 Docker 卷 `flashrt-pi05-models` 保存所有模型，删除或重建实验容器不会
删除该卷。假设宿主机已有 `/path/to/pi05_libero_finetuned_v044`：

```bash
docker volume create flashrt-pi05-models

docker run --rm \
  --mount type=volume,src=flashrt-pi05-models,dst=/workspace/models \
  --mount type=bind,src=/path/to/pi05_libero_finetuned_v044,dst=/source,readonly \
  flashrt:5090 bash -lc \
  'mkdir -p /workspace/models/pi05_libero_finetuned_v044 && \
   cp -a /source/. /workspace/models/pi05_libero_finetuned_v044/'
```

请把 `/path/to/pi05_libero_finetuned_v044` 替换为真实的绝对路径。不要把权重复制进 Git
仓库，也不要提交模型文件。

### 5.2 创建实验容器并下载其他模型

```bash
export ROBOT_BENCH_ARTIFACTS="$(pwd)-profile-artifacts-20260804"
scripts/robot_bench_container.sh start
```

第一次执行会自动完成：

- 创建并启动 `flashrt-robot-bench` 容器；
- 创建独立的 LeRobot Python 环境；
- 安装 LIBERO、EGL 和五模型所需依赖；
- 下载固定 revision 的 LIBERO assets；
- 下载 SmolVLA、GR00T N1.7、VLA-JEPA、MolmoAct2 及必要的内部 backbone；
- 对脚本中固定了 SHA-256 的主要权重进行校验。

这一步会下载大量文件并可能持续很久。所有文件保存在 Docker 卷中，下载中断后再次执行
同一命令即可继续。GR00T 的四个 suite 检查点占用空间最大。

查看状态并进入容器：

```bash
scripts/robot_bench_container.sh status
scripts/robot_bench_container.sh enter
```

进入后工作目录是 `/workspace/FlashRT`，模型目录是 `/workspace/models`。退出容器 shell
不会停止容器。需要暂停时可在宿主机执行：

```bash
docker stop flashrt-robot-bench
```

恢复时再次执行 `scripts/robot_bench_container.sh start`。

## 6. 运行五模型实验

以下命令均可直接在宿主机仓库根目录运行，脚本会自动启动并进入同一个容器。

### 6.1 六个代表动作的闭环测试

```bash
scripts/robot_bench_e2e.sh quick \
  /workspace/models/robot-bench-results/final_20260811/e2e
```

该协议运行五个模型、六个代表任务、每个任务 10 个 episode，共 300 个 episode。六个
任务是：

```text
libero_spatial:0
libero_object:0
libero_goal:0
libero_goal:7
libero_10:0
libero_10:9
```

完整四套件测试需要 2000 个 episode，耗时显著更长：

```bash
scripts/robot_bench_e2e.sh full \
  /workspace/models/robot-bench-results/full_20260811/e2e
```

### 6.2 稳态延迟和频率

```bash
scripts/robot_bench_latency.sh all \
  /workspace/models/robot-bench-results/final_20260811/latency
```

所有模型使用同一个 `libero_spatial:0`、seed 0 的真实初始观测。默认预热 10 次、测量
30 次，记录 P50/P90/P95 延迟、重规划频率、首次推理和峰值显存。也可只运行一个模型：

```bash
scripts/robot_bench_latency.sh pi05 \
  /workspace/models/robot-bench-results/final_20260811/latency
```

### 6.3 Nsight Systems 和 Nsight Compute

```bash
# 五模型：NSys 时间线 + NCU 代表性热点 kernel
scripts/robot_bench_nsight.sh all \
  /workspace/models/robot-bench-results/final_20260811/nsight full

# 只采集 NSys，速度更快
scripts/robot_bench_nsight.sh all \
  /workspace/models/robot-bench-results/final_20260811/nsight_systems systems
```

只分析单个模型时，把 `all` 替换为表格中的脚本标识。输出包括：

- `nsys.nsys-rep`：CPU/GPU 完整时间线；
- `nsys.sqlite`：可查询的 NSys 数据库；
- `nsys_stats_*.csv`：CUDA API 和 GPU kernel 汇总；
- `ncu.ncu-rep`：代表性热点 kernel 的 NCU 报告；
- `ncu_details.csv`：NCU 指标导出结果。

## 7. 保存和查看结果

运行结果默认位于 Docker 卷中的：

```text
/workspace/models/robot-bench-results/
```

把结果复制回项目报告目录：

```bash
docker cp \
  flashrt-robot-bench:/workspace/models/robot-bench-results/. \
  reports/robot_model_comparison_20260811/results/
```

报告入口：

- [实验复现说明与改动清单](reports/robot_model_comparison_20260811/README.md)
- [正确性与性能测试报告](reports/robot_model_comparison_20260811/robot_model_test_report.md)
- [Nsight 性能分析报告](reports/robot_model_comparison_20260811/robot_model_performance_analysis.md)
- [结果快照状态](reports/robot_model_comparison_20260811/results/README.md)

`.nsys-rep` 和 `.ncu-rep` 可分别用 Nsight Systems、Nsight Compute GUI 打开；不安装
GUI 时也可以直接查看仓库中导出的 CSV 和 JSON。

## 8. 只部署 FlashRT 推理环境

如果不运行 LIBERO 五模型对比，只需要 FlashRT 本身，完成第 4 节镜像构建后即可启动：

```bash
docker run --rm --gpus all --ipc=host -it \
  --mount type=bind,src=/absolute/path/to/checkpoints,dst=/checkpoints,readonly \
  flashrt:5090 bash
```

容器内的基本调用方式：

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/checkpoints/pi05",
    config="pi05",
    framework="torch",
)

actions = model.predict(
    images=[base_image, wrist_image],
    prompt="pick up the red block",
)
```

输入图像的格式、具体模型参数和其他硬件部署方式见英文
[`README.md`](README.md)、[`docker/README.md`](docker/README.md) 和
[`docs/INSTALL.md`](docs/INSTALL.md)。

## 9. 常见问题

### 找不到 `flash_rt_kernels*.so`

确认执行过第 4 节“导出机器人容器使用的扩展”，并检查：

```bash
find "$(pwd)-profile-artifacts-20260804/build" -maxdepth 1 -name '*.so' -ls
```

如果使用自定义目录，运行脚本前必须设置相同的 `ROBOT_BENCH_ARTIFACTS`。

### 找不到 Pi0.5 checkpoint

检查卷内路径：

```bash
docker run --rm \
  --mount type=volume,src=flashrt-pi05-models,dst=/workspace/models \
  flashrt:5090 \
  ls -lah /workspace/models/pi05_libero_finetuned_v044
```

目录名必须为 `pi05_libero_finetuned_v044`。

### 模型下载中断

重新执行：

```bash
scripts/robot_bench_container.sh start
```

脚本会检查已有文件并继续下载。可通过环境变量降低 GR00T Xet 下载并发：

```bash
ROBOT_BENCH_HF_XET_CONCURRENCY=16 \
  scripts/robot_bench_container.sh start
```

### CUDA 显存不足或机器正在被占用

先检查：

```bash
nvidia-smi
docker exec flashrt-robot-bench ps -eo pid,cmd
```

不要在其他训练或推理任务占用 GPU 时采集性能数据。正确性测试可以恢复后重新运行，性能
和 Nsight 数据必须在 GPU 空闲、功耗和频率状态稳定时重新采集。

### NCU 报告无权限

实验容器创建时已经加入 `SYS_ADMIN` capability。如果仍出现性能计数器权限错误，需要由
服务器管理员开放 NVIDIA GPU performance counters，然后重新创建或启动测试环境。

### 如何确认当前容器、镜像和卷

```bash
scripts/robot_bench_container.sh status
docker inspect flashrt-robot-bench --format '{{.Config.Image}}'
docker volume inspect flashrt-pi05-models
```

## 10. 数据与结论边界

- 不要把不同随机数协议、不同任务集合或不同硬件上的结果直接混在一张排名表里。
- 端到端成功率反映 checkpoint、预处理、动作转换和环境闭环的整体效果，不只反映网络
  结构本身。
- 推理频率不等于机器人真实控制频率；真实控制还包含相机、通信、环境或硬件执行时间。
- NCU 只分析代表性热点 kernel，必须结合 NSys 的完整调用链判断整体瓶颈。
- 当前归档包含历史 X-VLA 结果，仅用于说明其被排除的原因，不属于正式五模型对比。
