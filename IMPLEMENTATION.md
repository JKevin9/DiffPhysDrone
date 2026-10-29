# 双 profile 训练与 Newton 测试

实现日期：2026-10-27。训练入口仍为 `main_cuda.py`；用单一 `--profile` 选择完整的模型、输入与周期契约。

## 训练

在 DiffPhysDrone 根目录运行：

~~~bash
/home/user/miniconda3/envs/env_aerial/bin/python main_cuda.py --profile vanilla
/home/user/miniconda3/envs/env_aerial/bin/python main_cuda.py --profile Newton-Modifed
~~~

默认 50,000 iterations、batch 64、episode 10 s、seed 42。以上会启动长训练；本次只执行了短 smoke。
原有 newton 环境也通过了 Vanilla 小批量训练验证。没有安装或替换现有 CUDA 扩展：若缺少 quadsim_cuda，程序从原始源码构建独立的 diffphys_dynamics_cuda 缓存；场景查询使用独立 diffphys_scene_cuda。首次调用需要 NVCC/C++ 编译器和 ninja。

~~~bash
# 每 8×8 m tile 的柱体数量、半径、倾角等参数
python main_cuda.py --profile Newton-Modifed --scene-config configs/procedural_scene.json

# 短训练检查，不代表训练效果
python main_cuda.py --profile Newton-Modifed --batch-size 2 --num-iters 1 --episode-length-s 0.24

# 继续人工指定的 checkpoint；num-iters 是累计迭代数
python main_cuda.py --profile vanilla --resume /absolute/path/checkpoint_010000.pt --num-iters 50000
~~~

每个 run 保存 `run_config.json`、`metrics.jsonl`、TensorBoard 数据，以及默认每 1,000 次和结束时的 checkpoint。Checkpoint 包含版本、profile、输入配置、权重、已完成迭代、优化器和 scheduler；加载严格检查键、shape 和 finite 值。恢复采样从显式 seed 重新开始，不宣称逐位恢复原 run 的 RNG 状态。
`--video` 记录实际送入模型的深度输入，使用固定显示范围，默认最多 60 秒，可用 `--video-seconds` 修改。旧 args 文件仅保留为 Vanilla 的兼容别名；场景改为各环境独立生成，不再表示旧的 swarm 固定起终点任务。

## 输入与网络

| 项目 | vanilla | Newton-Modifed |
|---|---|---|
| 训练周期 | 1/15 s | 0.02 s |
| Benchmark 推理周期 | 67 × 1 ms | 20 × 1 ms |
| 图像 | 原相机投影 48×64 → max-pool 12×16 | 110×110 扫描 → 当前 yaw 重投影 80×90 |
| 卷积尺寸 | 32×6×8 → 64×4×6 → 128×2×4 | 内部池化 32×32；32×16×16 → 64×8×8 → 128×8×8 |
| 训练深度噪声 | 原 inverse-depth std=0.02 | AIRNet Gaussian 0.04 m、dropout 0.02、block 0.05、episode immunity 0.2 |
| 初始姿态扰动 | 保留 | 关闭 |

两版均为 10D 状态、6D interleaved 动作、192D GRU；state_proj / fc 初始化缩放保持 0.5 / 0.01。
状态顺序：yaw 系速度、限幅目标速度、原 Newton 的 `R[:, 2, :]`（世界 up 在 body 系的表示）、0.18 m 裕度。目标重合时避免除零。
动作按 `action.reshape(B, 3, 2).unbind(-1)` 拆为两个三维向量，变换至世界 ENU 后相减得到净加速度。
训练保留上一策略步动作进入动力学的队列语义；同一策略步的图像与状态统一使用执行动力学前的位姿。

Modified 扫描水平 ±110°、垂直 ±80°、10 Hz、最大射线距离 60 m；输出 HFOV 180° / VFOV 160°、equal-sin、0.15–40 m，空像素填 40 m。缓存世界点云每个策略步按当前 COM/yaw 重投影。
安装偏移为 link 系 (0,0,0.09) m；显式处理 Yunque calibration.yaml 的 link→COM 向量，避免把 link 当作 COM。深度变换为 `10 exp(-d) - 0.1d + 4`。Benchmark 关闭深度噪声。

## 场景与任务

默认有效区域 256×32 m，按 8×8 tile 配置密度；每个环境包含完整 z=0 地面、6–10 m roof、四面封闭墙。柱体是有界、可倾斜的解析圆柱，默认半径 0.1–0.3 m、倾角 0–60°。
浮动物体计数和尺寸/高度参数参照 aerial_lab 的 FOREST_UNIFORM_CFG / mesh_terrains.py：每 tile 为 floor(0.8 × 柱数)，box/sphere/ellipsoid 等概率；解析实体替代三角网格。柱数可设整数或 [min,max]。
起终点在生成完成的几何中检查边界、地面、roof 和障碍间隙；出生 AGL 0.5–5.5 m，默认净间隙 0.38 m。

远/中/近/混合概率 40/30/20/10；距离为 130–200、20–100、0.5–20、0.5–200 m；速度为 25–40、6–19、1–19、1–40 m/s。近距离排除完全重合的退化任务。
训练代码没有地图或固定 trials 读取入口，不读取 merged_1019 或其源地图。

## 人工选择 checkpoint 后测试

在 aerial_lab 根目录运行，参数保留计划中的拼写 `Newton-Modifed`：

~~~bash
# 只验证 checkpoint，不启动模拟器
/home/user/miniconda3/envs/env_aerial/bin/python scripts/competitor/newton/standard_test.py \
  --profile Newton-Modifed --checkpoint /absolute/path/checkpoint_050000.pt \
  --checkpoint-only --device cpu

# 少环境 smoke；newton_run.json 始终标记 complete=false
/home/user/miniconda3/envs/env_aerial/bin/python scripts/competitor/newton/standard_test.py \
  --profile Newton-Modifed --checkpoint /absolute/path/checkpoint_050000.pt \
  --num-envs 2 --headless --smoke-steps 10

# 完整固定任务，不自动选择、比较或重跑 checkpoint
/home/user/miniconda3/envs/env_aerial/bin/python scripts/competitor/newton/standard_test.py \
  --profile Newton-Modifed --checkpoint /absolute/path/checkpoint_050000.pt \
  --num-envs 50 --headless
~~~

Vanilla 改用 `--profile vanilla`；非默认仓库位置传 `--diffphys-root /absolute/path/DiffPhysDrone`。
固定加载 standard_benchmark_1019.yaml 和 merged_1019/trials.csv；可显式指定 `--trials`，速度筛选使用 `--vel-range MIN MAX`。
原始无元数据 state_dict 不被猜测为任一 profile，需使用新版训练的自描述 checkpoint。

Newton 专用 ActionTerm 在 1 ms 物理步内调度推理，保持环境 decimation=20 及现有任务生命周期；Vanilla 不改变其他环境时序。沿用 ResultTracker、LoadedPositionCommand 和 recording 的过滤、跳过及完成规则。
GRU 和保持动作按环境 reset；reset 后首次推理的动作及 hidden 清零。Smoke 另核对 ENU→NED 的实际加速度输入与部分环境 reset。

加速度进入已有 aerial_core position 模式的纯加速度接口及电机/动力学链路，不新增另一套控制器。前置检查要求已对齐的 thr_hover=0.3532、tiltmax_air=80°。
原动力学 CUDA 文件、控制器配置未改写，Env 中并行工作给出的 12.32 / 4.33 中心值保留。

每个独立 run 的 `newton_run.json` 记录 checkpoint/config/trials 路径和 SHA-256、profile、推理周期、CSV 位置与完成状态；记录目录提供 trials.csv 和 trajectory.csv。
完整测试验证记录行数、唯一 trial ID、跳过计数及非空轨迹。Smoke 未结束 trial 时仅产生带表头的空 trials.csv，不补造结果。

## 验证与边界

~~~bash
# DiffPhysDrone 根目录
OMP_NUM_THREADS=2 /home/user/miniconda3/envs/env_aerial/bin/python -m pytest tests/test_contracts.py -q

# aerial_lab 根目录
OMP_NUM_THREADS=2 /home/user/miniconda3/envs/env_aerial/bin/python -m pytest \
  tests/test_newton_runner_contracts.py tests/test_diffphys_alignment.py -q
~~~

测试覆盖 checkpoint、状态与动作、20/67 步调度、GRU reset、相机预处理、扫描缓存、场景约束、速度分布和 CSV 计数。
AIRNet 对照直接使用当前 lidar_pattern、RayCaster core、PC2DepthImgConverter、reward_like 和噪声实现：相同六面封闭场景、含非零 roll/yaw 的位姿及扫描间运动重投影。点云绝对容差 3e-5 m、深度绝对容差 5e-5 m；投影/变换和固定 RNG 噪声的独立对照要求逐元素一致。

本次只做本地单元测试、小批量短训练和少环境短仿真，未启动 50,000 次训练，也未完成 1,200 条固定任务的正式评测。正式成绩仍须由用户人工选定训练 checkpoint 后生成；smoke 不代表导航成功率或性能结论。


## 本次本地验证记录

2026-10-27：15 项 DiffPhys 契约测试、3 项 Newton CSV 测试和 3 项已有 dynamics-alignment 回归，共 21 项通过；Ruff、格式、compileall、两仓库 git diff --check 通过。
两种 profile 各执行 batch=2、1 iteration 的短训练并重新加载 checkpoint。最终 Isaac smoke：Vanilla 2 环境 × 9 个环境步，每环境推理 3 次，trajectory.csv 18 行；Modified 2 环境 × 6 个环境步，每环境推理 6 次，trajectory.csv 12 行。两者均验证实际控制器加速度映射和部分 reset；trials.csv 均为 0 条数据行，明确不代表已完成正式试验。
结构化证据见 outputs/implementation_validation_20261027.json；短 run 的原始日志与 CSV 路径在该文件中。未执行 git commit/push。
