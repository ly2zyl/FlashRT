# 五机器人模型 LIBERO 正确性与性能测试报告

测试日期：2026-08-11
测试环境：LIBERO / MuJoCo，headless EGL
硬件：NVIDIA GeForce RTX 5090（SM120，32 GB）
容器：`flashrt-robot-bench`（镜像 `flashrt:5090`）

## 1. 报告目的

本报告回答两个问题：

1. 模型在相同机器人环境里是否真的能完成动作，而不只是输出形状正确的张量；
2. 在完成质量之外，各模型的控制频率、完整重规划延迟、尾延迟和显存成本如何。

正式对比模型为 Pi0.5、SmolVLA、GR00T N1.7、VLA-JEPA 和 MolmoAct2。X-VLA 已从
正式组移除，因为当前集成在 60 个代表性 episodes 中为 0/60，同一用例用官方
`lerobot-eval` 交叉检查仍失败。这里的结论是“当前 checkpoint/环境组合不可用”，
而不是证明 X-VLA 架构在任何训练条件下都无法完成机器人动作。

## 2. 测试用例

| 列简称 | Suite / task | 动作内容 | 最大步数 | 覆盖能力 |
|---|---|---|---:|---|
| 空间抓放 | `libero_spatial:0` | 把盘子和 ramekin 之间的黑碗放到盘子上 | 280 | 空间关系、抓取、放置 |
| 物体抓放 | `libero_object:0` | 把 alphabet soup 放入篮子 | 280 | 物体识别、抓取、容器放置 |
| 开抽屉 | `libero_goal:0` | 打开柜子中间抽屉 | 300 | 接触操作、沿约束方向运动 |
| 开炉灶 | `libero_goal:7` | 打开炉灶 | 300 | 小目标定位、接触式开关 |
| 双物体搬运 | `libero_10:0` | 把 alphabet soup 和 tomato sauce 都放入篮子 | 520 | 两个连续子目标、状态切换 |
| 微波炉复合动作 | `libero_10:9` | 打开微波炉、放入杯子并关门 | 520 | 长序列、门体交互、误差恢复 |

每个模型对每个动作运行 10 次，环境使用 seed 0–9，共 `5×6×10=300` 个 episodes。
每个模型进程启动时固定一次策略随机流，并按同一用例顺序连续运行；成功由 LIBERO
环境的 `is_success` 判定，不使用人工观看或模型自评。

五个模型接收同一个环境的两路 256×256 相机观测、语言指令和机器人状态，并最终向
环境提交相同含义的 7 维相对动作。模型内部保留各自原生 action chunk、去噪步数、
数值精度和预处理，这反映现成 checkpoint 的实际部署成本。

## 3. 如何阅读性能指标

- **成功率**：成功 episodes / 总 episodes，是正确性的主指标。
- **控制频率（Hz）**：环境总步数 / 总墙钟。它包含预处理、动作选择和 MuJoCo step，
  但模型会缓存 action chunk，因此很多控制 step 不需要重新运行模型。
- **重规划频率（Hz）**：每秒能生成多少个完整 action chunk。它不包含 MuJoCo step，
  更适合比较纯策略部署速度。
- **P50**：一半重规划比该值快，代表典型延迟。
- **P95**：95% 重规划比该值快，反映偶发卡顿和尾延迟。
- **峰值显存**：单模型进程从加载到稳态测量的 PyTorch peak allocated；不等于
  `nvidia-smi` 中包含驱动上下文的进程总占用。
- **完成步数**：只对成功 episodes 求均值。越少通常表示路径更直接，但不同 chunk
  长度会影响闭环纠错节奏，不能单独当作模型质量排名。

控制频率高不等于动作做得好：一个模型可以很快地重复错误动作直到超时。因此报告
先看成功率，再在成功质量可接受的候选中比较延迟和显存。

## 4. 正确性主结果

<!-- FINAL_CORRECTNESS_TABLE -->

## 5. 六组闭环性能

<!-- FINAL_CONTROL_FREQUENCY_TABLE -->

<!-- FINAL_COMPLETION_STEPS_TABLE -->

## 6. 独立重规划性能

性能输入取自六组正式实验中的 `libero_spatial:0`、seed 0 真实初始观测。每个模型
warmup 10 次，正式测量 30 次；计时包含该模型完成一次 replan 所需的预处理和数据搬运。

<!-- FINAL_LATENCY_TABLE -->

## 7. 按动作类型分析

<!-- FINAL_TASK_ANALYSIS -->

## 8. 按模型综合分析

<!-- FINAL_MODEL_ANALYSIS -->

## 9. 正确性检查

正式结果必须同时满足：

- 进程无异常退出，所有模型输出均为有限数值；
- 原生动作块最终转换为环境所需的 7 维连续动作；
- 双相机、任务文本、状态、控制模式和 seed 对齐；
- 成功来自环境 `is_success`，失败 episode 正常运行至成功或最大步数；
- GR00T 在纳入正式组前，已经用 Spatial 官方 checkpoint 完成过一次真实环境 1/1
  冒烟验证，而不是只检查张量 shape；
- 模型 revision 和大权重 SHA-256 固定，运行时启用离线模式，避免测试中静默换权重。

## 10. 统计解释和局限

1. 每个动作 10 次比单次 episode 稳定，但每模型 60 次仍是代表性筛查，不是完整
   LIBERO 40-task、400-episode 标准评测。
2. LeRobot 文档指出完整结果可能随评测 seed 波动数个百分点，并建议多 seed 平均。
   本轮固定 seed 0–9，便于模型间配对比较，但没有再重复三组外层 seed。
3. GR00T 使用官方 suite 专用 checkpoint；其他模型也使用各自公开/本地 LIBERO
   checkpoint。训练数据、训练预算和量化方式不同，因此这是“可部署策略成品”比较，
   不是只控制网络结构变量的消融实验。
4. 闭环控制频率受 action chunk 缓存和环境步耗时影响；重规划频率不包含环境。两者
   回答不同问题，不能互相替代。
5. Pi0.5 使用 FlashRT FP8 路径；若其成功率低于参考实现，需要再做 BF16/reference
   闭环 A/B，区分模型能力、量化校准与当前运行时的影响。

LeRobot 的[官方 LIBERO 文档](https://huggingface.co/docs/lerobot/libero)规定四个标准
suite、双相机、8 维状态、7 维动作，并推荐每 task 10 episodes；本报告明确标为六动作
代表集。GR00T 的[官方 LeRobot 文档](https://huggingface.co/docs/lerobot/groot)给出的
初步结果是 Spatial 95%、Object 100%、Goal 98%、LIBERO-10 93%，每 suite 至少 50
episodes。这些数字只用于外部合理性检查，不与本地 60-episode 样本混算。

## 11. 原始结果与复现

原始 JSON、NSys/NCU 文件、进入容器和一键命令见同目录 `README.md`。正式原始结果：

```text
/workspace/models/robot-bench-results/final_20260811/
```

宿主机归档：

```text
/home/zhangyunli/FlashRT-profile-artifacts-20260804/robot_profile/final_20260811/
```
