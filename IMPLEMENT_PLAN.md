# DiffPhysDrone → Newton benchmark 实施计划

日期：2026-10-26  
状态：2026-10-27 已实施双 profile 训练和 Newton benchmark 适配，并执行本地验证；长训练与人工选定 checkpoint 的完整评测尚未执行。运行说明见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 目标与边界

在 DiffPhysDrone 提供两个可切换的训练 profile，并让训练完成后由用户人工选择 checkpoint，再通过 aerial_lab 的 Newton benchmark 入口跑现有固定测试任务，最终每个被测模型保存 trials.csv 与 trajectory.csv。

训练不得读取 merged_1019 或其六张源地图；merged_1019 与固定 trials 仅用于最终 benchmark。无需实现 choose_checkpoint，也不自动重跑或比较候选 checkpoint。动力学和加速度控制器对齐由并行工作负责，本计划将其视为 benchmark 前置接口，不改写相关文件。

## 已确认的模型与运行契约

| 项目 | vanilla | Newton-Modifed |
|---|---|---|
| 传感输入 | 深度相机，保留原始 FOV；渲染 48×64，池化 4 倍为 12×16 | AIRNet 当前输入路径：约 10 Hz、110×110 yaw-only 点云；每个 50 Hz 策略步用最新扫描按当前 yaw 重投影为 80×90 深度图 |
| 深度图几何 | 保留 DiffPhysDrone 原有相机投影与深度预处理；FOV 使用原配置值 0.82 | HFOV 180°、VFOV 160°、equal-sin 垂直采样；对齐当前 AIRNet 的 yaw-only、重投影及 reward_like 变换 |
| 策略周期 | 训练 1/15 s；benchmark 物理步长 1 ms 时每 67 步推理一次，约 14.93 Hz | 训练及 benchmark 均为 1/50 s，即 1 ms 物理步长下每 20 步推理 |
| 网络 | 指定的 Vanilla CNN + GRU；输入图像为 12×16 | 先将 80×90 池化为 32×32，再使用指定 CNN + GRU；按已选 A 方案配置 padding/stride，使卷积输出为 128×8×8，线性层输入维度据此设定 |
| 随机化 | 保留原有 Vanilla 随机化 | 不做姿态扰动 |

两种模型统一采用 benchmark 所需的 10 维状态输入和 6 维动作输出，隐藏维度 192。状态为 yaw 坐标系速度 3 维、限幅目标速度 3 维、机体 z 轴方向 3 维、安全裕度 1 维；目标速度按目标距离与 trial 设定速度限幅，安全裕度为 0.18 m。6 维动作沿用 Newton benchmark 约定，拆为两个 3 维向量供加速度接口计算目标加速度。示例构造函数中的 9/4 默认值不作为 benchmark 接口。

两个 profile 都覆盖 1–40 m/s，默认训练 50,000 iterations。通过单一 profile 选项切换 vanilla 与 Newton-Modifed；checkpoint 保存模型权重和 profile/输入配置元数据。训练结束后用户显式提供选定的 checkpoint 给 benchmark。

## DiffPhysDrone 训练实现阶段

1. 在 main_cuda.py 增加 profile 选择与集中配置解析。配置中固定模型、输入预处理、控制周期、速度范围和随机化选项，避免通过多个 args 文件切换时遗漏参数。保留现有训练入口和损失流程。
2. 将 vanilla 与 Newton-Modifed 网络分离为清晰可选的模型定义，保持给定层序、LeakyReLU(0.05)、192 维 GRUCell 及初始化缩放。验证各 profile 前向输出分别为 6 维，Modified 的卷积/线性张量尺寸连通。
3. vanilla 保持 48×64 相机深度和原始 FOV，降采样到 12×16；保留现有深度裁剪、变换和噪声行为，以及已有随机化；从两个 profile 的 CLI、配置和场景路径中彻底移除 random_rotation 参数及代码。
4. Newton-Modifed 按 AIRNet 输入契约生成当前 yaw 下的 80×90 深度观察；遵从 110×110、约 10 Hz 的扫描更新节奏，在两次扫描间复用最近点云并随当前 yaw 重投影。使用 0.15–40 m 范围及 reward_like 参数 exp_scale=10、exp_decay=1、linear_scale=0.1、offset=4；关闭姿态扰动。
5. 替换固定起终点/旧地图式训练场景为独立程序化生成器。每个环境的地面为 z=0 连续平面且无洞；roof 概率 100%，高度 6–10 m，覆盖整个有效飞行区域及所有合法起终点；出生高度 AGL 为 0.5–5.5 m；以 voxel 或等价几何实现封闭边界墙。
6. 在训练配置中提供每个 8×8 区域的柱体数量（固定值或随机区间）、半径范围和倾角范围；浮动物体的数量与分布以 aerial_lab 当前 FOREST_UNIFORM_CFG 地形生成器为参照。所有合法起点和终点须通过边界、地面、roof 与障碍间隙检查。场景生成不依赖 merged_1019、其源地图或固定 benchmark trials。
7. 任务采样参考 aerial_lab 的 ALL_SPEED_POSITION_PATTERNS：保留远/中/近/混合任务类别与大致 40/30/20/10 概率，调整速度子区间使总体覆盖 1–40 m/s；训练自行采样起终点，不复用 benchmark 固定试验。
8. 每个训练 run 记录 profile、随机种子、传感参数、控制周期、场景参数及速度分布；定期保存可直接加载的 checkpoint。训练仍由用户按需启动，本计划阶段不启动长训练。

## aerial_lab Newton benchmark 适配阶段

1. 以 commit d889ef69f83bd9713f87ca641f96a68df18f09ca 的 scripts/competitor/newton/standard_test.py 为适配参考，保留当前 standard_test 的任务生命周期和结果追踪。增加明确的 profile 与 checkpoint 参数；不做自动 checkpoint 选择。
2. 使用 standard_benchmark_1019.yaml 和当前固定 trials 加载评测任务。只在测试中使用 merged_1019；沿用当前速度筛选、无效/跳过试验处理及完成条件。
3. 为两个 profile 各自构造正确的图像输入。Modified 逐项匹配 AIRNet：传感器模式、yaw-only 坐标、110×110 扫描与更新周期、80×90 输出、180°×160° FOV、equal-sin 投影、深度范围和 reward_like 参数；benchmark 关闭深度噪声。Vanilla 使用其 48×64 相机输入、FOV 和 12×16 池化契约。
4. 对齐 10 维状态、6 维动作、目标速度限幅、0.18 m 安全裕度、GRU hidden state 生命周期及 reset 后清零/首步动作处理。推理结果按 Newton 的两个三维向量约定换算为目标加速度，并交给并行工作提供的 aerial_core 加速度控制配置。
5. 控制周期按 profile 生效：Modified 每 20 个 1 ms 物理步更新；vanilla 每 67 步更新并保持动作。不得为 vanilla 改变 benchmark 的物理步长或其他环境时序。
6. 启动前对 checkpoint 键名、profile 元数据、输入输出 shape、设备和 finite 值做 fail-fast 校验。使用短时少环境 smoke test 确认模型可加载、输入可前向、reset 后 hidden state 清零、控制器收到预期动作；之后才运行完整固定 trials。
7. 每个最终测试 run 输出独立目录，至少包含 trials.csv 与 trajectory.csv；验证 trial 行数、轨迹记录非空、结果可由现有工具读取，并把所用 checkpoint、配置和 CSV 路径留档。最终 CSV 由人工选定 checkpoint 对应的 benchmark run 产生。

## 验收标准

- profile 选择能分别启动 vanilla 和 Newton-Modifed；模型权重互不混淆，训练/benchmark 均明确记录 profile。
- 两版速度采样范围均为 1–40 m/s；Vanilla 图像 shape 为 1×12×16，Modified 输入路径最终 shape 为 1×80×90 且模型内部池化为 32×32。
- Modified 的 AIRNet 输入可在同一场景、同一位姿下与 aerial_lab 观察管线逐项比对：射线/点云坐标语义、yaw 投影、图像方向、深度值、无回波填充值、深度变换和策略更新时序一致。
- 场景检查证明每个环境都有 z=0 地面、roof 始终存在且高度在 6–10 m、roof 覆盖合法起终点、起点高度在 0.5–5.5 m、边界封闭；柱体密度参数可固定或随机配置，浮动物体分布与 benchmark 基准参数一致。
- Benchmark 对固定 trials 完成运行且按既有规则处理过滤/跳过项，产出可读的 trials.csv 和 trajectory.csv；不把 smoke test 或有限推理当作完整 benchmark 结果。
- Dynamics/controller 配置已由并行工作提供并通过接口核验；否则最终闭环 benchmark 阶段标记为等待该依赖。

## 主要代码落点与参考

- DiffPhysDrone：main_cuda.py、model.py、env_cuda.py，以及新增/调整的训练配置和场景生成模块。
- aerial_lab：scripts/competitor/newton/standard_test.py、aerial/aerial/config/helpers/lidar_mode_params.py、aerial/aerial/mdp/observations/basic_observation.py、aerial/aerial/utils/converter.py、aerial/aerial/config/tasks/command_patterns.py、aerial/aerial/config/tasks/lidar_navigation/specific_config/standard_test_cfg.py、aerial/aerial/config/global_cfg/yamls/standard_benchmark_1019.yaml、aerial/aerial/recording。
- benchmark 地形分布参照 aerial/aerial/terrains/registered_terrain.py 与 aerial/aerial/terrains/sub_terrain/；不把测试地形数据用于 DiffPhysDrone 训练。

