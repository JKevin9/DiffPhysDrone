# 从 `5a96a9fd` 读到当前代码

基线是最初的单脚本训练：`main_cuda.py` 里随机球 / 柱 / 体素，深度相机，约 15 Hz。现在仍是这条可微物理回路，但用 `--profile` 切两套输入契约，场景换成独立的解析森林，并对齐 Newton benchmark 的 10 维状态和 6 维动作。更全的运行说明在 `IMPLEMENTATION.md`。

## 先看这几个文件

| 你原来找的地方 | 现在 |
|---|---|
| `main_cuda.py` 的训练循环 | 仍在这里，核心是 `rollout_loss` |
| `model.py` | 只是别名。网络在 `diffphys/models.py` |
| `env_cuda.py` 里生成球、柱、体素 | 删了。几何在 `diffphys/scenes.py`，射线和最近点在 `src/scene_kernel.cu` |
| `env_cuda.py` 的 `render` / `run` | `observation()` 出图；`run()` 仍走原来的动力学 CUDA |
| 一堆场景开关 | 全部收进 `diffphys/profiles.py` 的 `vanilla` 和 `Newton-Modifed` |

`diffphys/runtime.py` 不参与训练。它是给 aerial_lab 的 1 ms 物理步用的：vanilla 每 67 步推理一次，Modified 每 20 步一次，reset 后丢掉第一步动作和 GRU。

## 训练回路里没变的部分

损失项还是那一组：速度跟踪、速度预测、bias、加速度 / jerk / snap、避障、碰撞、速度、贴地。避障仍是间隙的二次惩罚，碰撞仍是 `softplus(-32 * distance)`，都乘接近速度。动作仍延迟一个策略步再送进动力学。优化器仍是 AdamW + cosine，`grad_decay=0.4`。

6 维动作的拆法与原来 `reshape(B, 3, -1)` 相同，是交错的两个三维向量，变到世界系后相减，再套推力估计误差和重力。GRU 仍是 192 维；`state_proj` / `fc` 仍乘 0.5 / 0.01。

`src/dynamics_kernel.cu` 的前向、反向和姿态更新是同一套公式，diff 主要是排版。训练不再调用 `quadsim_kernel.cu` 的球柱体素渲染。没装 `quadsim_cuda` 时，会从这三份源码 JIT 出一个独立扩展，不覆盖已安装的模块。

## 必须知道的行为变化

**1. 观测和状态改到同一步、动力学之前。**  
旧循环先 `render`，再 `env.run`，然后用**新**位姿组 state。现在图像和 10 维状态都取自动力学之前的位姿，再执行上一步动作。

**2. 周期固定，不再抖动。**  
旧代码每个步 `normalvariate(1/15, 0.1/15)`，默认 150 步。现在按秒数切：默认 10 s。vanilla 固定 1/15 s（150 步），`Newton-Modifed` 固定 0.02 s（500 步）。

**3. 状态坐标系换了，维度没换（有里程计时本来就是 10 维）。**  
速度和限幅目标速度改到由 `atan2` 取出的 yaw 系，不再用“机体前向压到水平面”造的轴。第三段仍是 `R[:, 2]`（世界上方向在机体系）。裕度固定 0.18 m。`--no_odom` 没有了。

**4. 球和圆柱最近点的长度 bug 已修。**  
旧 `quadsim_kernel.cu` 先算出表面间隙 `dist`，再乘上没有单位化的 `(c - o)`。避障和碰撞用的向量长度变成 `dist * ||c - o||`。体素和地面仍是真实最近点，球、柱和盒子的距离尺度不一致。训练不再走这个核。`find_vec_to_nearest_pt()` 改走 `scene.closest()`：`src/scene_kernel.cu` 里球按椭球投影，有限圆柱把径向缩到半径再夹高度，向量长度就是到表面的距离。`quadsim_kernel.cu` 里的旧公式还在，训练不调用。

**5. 动力学按 Yunque 对齐。这是两套独立标定，不是把整套物理换成 aerial-core。**  
旧的 `speed_mtp` 和按最大速度放大障碍都去掉了。平移和偏航系数、阻力系数来源不同，不要混成一次拟合。

平移和偏航来自 2026-10-26 的短时阶跃，记录在 [`newton_diffphys_dynamics_alignment_results.md`](/home/user/project/aerial_stack/python_repos/aerial_lab/docs/newton_diffphys_dynamics_alignment_results.md)。先冻结当前 Yunque（`calibration=default`，质量 1.542 kg，风、噪声和随机化关闭），再对 aerial-core 的实际姿态和桨推力做阶跃：X/Y 为 ±2、±5 m/s²，Z 为 ±2 m/s²，yaw 为 ±10°、±30°。DiffPhys 只拟合已有的一阶更新 `alpha = exp(-k·dt)`。平移在上升段识别 `pitch_ctl_delay`，偏航保留原来的前向混合后再识别 `yaw_ctl_delay_inv`。不拟合增益或纯延迟，也不把曲线平移对齐。写入中心是 12.32 和 4.33，仍保留原来的 10% 高斯。这是名义响应近似：yaw 汇总 RMSE 大约下降 10.7%，水平通道和原来的 12 s⁻¹ 已经很接近。超调、X/Y 差异和更快的竖直响应留在残差里。

阻力没有进这次阶跃拟合。那份结果里阻力仍是附加项，主比较的是滤波后的 `act`。`drag_2` 是另一次投影：把关闭随机化后的 Yunque 机身阻力和四桨 H 力，投到现有的分轴公式上。形式不能改，只能定三个数：

```text
a_x = c_q v_x|v_x| + c_l v_x
a_y = c_q v_y|v_y| + c_l v_y
a_z = z_drag_coef * (c_q v_z|v_z| + c_l v_z)
```

机身是 `F = -½ρ CdA |v| v`，`CdA = exp(-4.2393 - 0.06036 u_z² - 0.04967(u_x² - u_y²))`，`ρ = 1.225`，质量 1.542 kg。四桨 H 力只作用在机体 xy；悬停转速取 SoC 0.875 的静平衡，约 1778 rad/s，对应一次系数约 0.183 s⁻¹，平飞再按 `ω = ω_hover √(T/(mg))` 提高转速。样本覆盖 1–40 m/s 的平飞（约 ±15° 侧滑）、小迎角前飞和纯轴向。损失是阻力加速度的均方误差，三类权重 0.5 / 0.3 / 0.2，在 `c_q`、`c_l`、`z_drag_coef` 上网格搜索。结果固定为 `drag_2 = [0.0041, 0.243]`、`z_drag_coef = 0.5`，不再按环境随机，也不再把二次项清零。前进比带来的推力损失沿推力轴，这三项表示不了，没有放进去。

**6. 场景是每个环境各自采样的封闭走廊，不读地图，也没有 swarm。**  
默认区域 256×32 m，8 m 一格。每个环境都有 z=0 地面、屋顶、四面墙，加上可倾斜圆柱和悬浮 box/sphere。起终点要躲开障碍，出生高度 0.5–5.5 m。任务按 40/30/20/10 抽远/中/近/混合，速度覆盖 1–40 m/s。不传 `--scene-config` 时用 `SceneConfig` 的代码默认值（每格 10 根柱、三种悬浮物）。`configs/procedural_scene.json` 只有显式传入才生效。

**7. 旧 checkpoint 不能加载。**  
新文件带 `format_version`、profile 和输入配置，键和 shape 对不上就拒绝。`--resume` 恢复权重、优化器和 scheduler；`--num-iters` 是累计迭代，采样从给定 seed 重来，不恢复旧 RNG。日志在 `logs/procedural/<时间>_<profile>/`。

贴地项顺手改对了：旧代码把 `coef_ground_affinity` 当成常数加进去，损失本身一直全额计入。现在是系数相乘，默认 0。

## 两个 profile

| | vanilla | Newton-Modifed |
|---|---|---|
| 训练周期 | 1/15 s | 1/50 s |
| 图像 | 48×64 相机，池化成 12×16；编码仍是 `3/d - 0.6` | 10 Hz、110×110 扫描，按当前 yaw 重投影成 80×90 |
| 网络 | 原来的三层卷积 | 先池化到 32×32，再另一套卷积，线性层吃 `128×8×8` |
| 深度噪声 | 编码后高斯 0.02 | 米制高斯 / dropout / 方块；benchmark 关闭 |
| 姿态扰动 | 保留（相机角、FOV、初值速度） | 关闭 |
| 深度 FOV | half-tan 0.82，相机俯仰 20°，与旧 `single_agent.args` 一致 | 水平 180°、垂直 160°，0.15–40 m |

Modified 的扫描装在 link 系 `(0,0,0.09)` m，并减去标定里的 link→COM，不要把 link 当质心。点云缓存在世界系，两次扫描之间只重投影。

## 工作区里还没提交的两处数值

相对最近一次提交 `23a7856`：

- `main_cuda.py` 的默认损失系数改成 `v_pred=0`、`collide=7.5`、`obj_avoidance=3.0`。后两个就是旧 `single_agent.args` 里的数；速度预测损失默认关掉，旧默认是 2。
- `configs/procedural_scene.json` 的 `poles_per_tile` 从 `[5, 12]` 改成 `[2, 10]`。这份 json 还把屋顶收成 6–8 m、悬浮物只有 box/sphere、数量系数 1.2。代码默认值更密，而且不自动读这个文件。

## 不用再找的开关

`--single`、`--gate`、`--scaffold`、`--ground_voxels`、`--random_rotation`、`--yaw_drift`、`--no_odom`、`--timesteps`、`--speed_mtp`、光流、多机互碰。`configs/single_agent.args` 和 `multi_agent.args` 现在只剩 `--profile vanilla`。

```bash
python main_cuda.py --profile vanilla
python main_cuda.py --profile Newton-Modifed --scene-config configs/procedural_scene.json
```
