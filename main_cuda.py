from collections import defaultdict
import math
from random import normalvariate  # 生成正态分布随机数
from matplotlib import pyplot as plt  # 绘图库
from env_cuda import Env  # 自定义CUDA加速的环境
import torch
from torch.nn import functional as F  # 神经网络函数工具
from torch.optim import AdamW  # 优化器
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度器
from torch.utils.tensorboard import SummaryWriter  # 训练过程可视化
from tqdm import tqdm  # 进度条工具
import os
from datetime import datetime
import argparse
from model import Model  # 自定义神经网络模型
import numpy as np
import cv2

# 解析命令行参数
parser = argparse.ArgumentParser()
# 模型相关参数
parser.add_argument("--resume", default=None)  # 预训练模型路径
parser.add_argument("--batch_size", type=int, default=64)  # 批大小
parser.add_argument("--num_iters", type=int, default=50000)  # 训练迭代次数
parser.add_argument("--run_name", type=str, default="your_run_name")  # 运行名称（用于TensorBoard记录）
parser.add_argument("--exp_name", type=str, default="your_exp_name")  # 运行名称（用于TensorBoard记录）

# 损失函数权重系数
parser.add_argument("--coef_v", type=float, default=1.0, help="smooth l1 of norm(v_set - v_real)")
parser.add_argument("--coef_speed", type=float, default=0.0, help="legacy")  # 遗留参数
parser.add_argument("--coef_v_pred", type=float, default=2.0, help="mse loss for velocity estimation (no odom)")
parser.add_argument("--coef_collide", type=float, default=2.0, help="softplus loss for collision")
parser.add_argument("--coef_obj_avoidance", type=float, default=1.5, help="quadratic clearance loss")
parser.add_argument("--coef_d_acc", type=float, default=0.01, help="control acceleration regularization")
parser.add_argument("--coef_d_jerk", type=float, default=0.001, help="control jerk regularizatinon")
parser.add_argument("--coef_d_snap", type=float, default=0.0, help="legacy")  # 遗留参数
parser.add_argument("--coef_ground_affinity", type=float, default=0.0, help="legacy")  # 遗留参数
parser.add_argument("--coef_bias", type=float, default=0.0, help="legacy")  # 遗留参数
# 训练参数
parser.add_argument("--lr", type=float, default=1e-3)  # 学习率
parser.add_argument("--grad_decay", type=float, default=0.4)  # 梯度衰减系数
# 环境/传感器参数
parser.add_argument("--speed_mtp", type=float, default=1.0)  # 最大目标速度倍数
parser.add_argument("--episode_length_s", type=int, default=10)  # 每个episode的时间步长
parser.add_argument("--ctl_dt", type=float, default=1 / 50)  # 每个step的时间步长
# 环境配置标志
parser.add_argument("--elevation_min", type=float, default=-90)  # 垂直视场角下限
parser.add_argument("--elevation_max", type=float, default=90)  # 垂直视场角上限
parser.add_argument("--azimuth_min", type=float, default=-180)  # 水平视场角下限
parser.add_argument("--azimuth_max", type=float, default=180)  # 水平
parser.add_argument("--single", default=False, action="store_true")  # 单一agent训练模式
parser.add_argument("--gate", default=False, action="store_true")  # 门形障碍开启与否
parser.add_argument("--ground_voxels", default=False, action="store_true")  # 使用地面体素
parser.add_argument("--scaffold", default=False, action="store_true")  # 脚手架模式
parser.add_argument("--random_rotation", default=False, action="store_true")  # 随机旋转
parser.add_argument("--yaw_drift", default=False, action="store_true")  # 偏航漂移模拟
parser.add_argument("--no_odom", default=False, action="store_true")  # 不使用里程计
parser.add_argument("--video", default=False, action="store_true")  # 使用相机

args = parser.parse_args()

# 初始化TensorBoard记录器
# specify directory for logging experiments
log_root_path = os.path.join("logs", args.exp_name)
log_root_path = os.path.abspath(log_root_path)
print(f"[INFO] Logging experiment in directory: {log_root_path}")
log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.run_name:
    log_dir += f"_{args.run_name}"
log_dir = os.path.join(log_root_path, log_dir)
writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
print(args)  # 打印参数配置
device = torch.device("cuda")

env = Env(
    batch_size=args.batch_size,
    width=240,
    height=120,  # 深度图分辨率
    # 64,
    # 48,  # 深度图分辨率
    azimuth_min=args.azimuth_min,
    azimuth_max=args.azimuth_max,
    elevation_min=args.elevation_min,
    elevation_max=args.elevation_max,
    grad_decay=args.grad_decay,
    device=device,
    single=args.single,
    gate=args.gate,
    ground_voxels=args.ground_voxels,
    scaffold=args.scaffold,
    speed_mtp=args.speed_mtp,
    random_rotation=args.random_rotation,
)
if args.no_odom:
    model = Model(7, 6)  # 无里程计: 7维状态输入
else:
    model = Model(7 + 3, 6)
model = model.to(device)

# 加载预训练模型（如果指定）
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)

# 初始化优化器和学习率调度器
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)  # 余弦退火调度


# 用于平滑记录训练指标
scaler_q = defaultdict(list)

if args.video:
    fps = round(1 / args.ctl_dt)
    width = 240
    height = 150
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或使用 'XVID' 生成AVI
    video_writer = cv2.VideoWriter("depth_video.mp4", fourcc, fps, (width, height), isColor=False)


def record_video(depth_frames):
    # 3. 创建视频写入器
    depth_frames = depth_frames.cpu().numpy()
    # 1. 归一化并转换为uint8
    normalized = (depth_frames - 0.3) / (24 - 0.3 + 1e-8) * 255
    normalized = normalized.astype(np.uint8)
    video_writer.write(normalized)


def smooth_dict(ori_dict):
    """将当前指标值添加到平滑队列"""
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))


def show_depth_image(depth_img, fig_size=(18, 6), font_size=14):
    # 创建可视化画布
    plt.figure(figsize=fig_size)
    plt.suptitle(f"Range Image", fontsize=font_size)
    plt.subplot(111)
    # 获取深度图的numpy数组并旋转180度调整方向
    depth_img = depth_img.clamp(0.3, 24)
    depth_np = depth_img.cpu().numpy()
    min_value = depth_np.min()
    max_value = depth_np.max()
    plt.imshow(depth_np, cmap="jet", vmin=min_value, vmax=max_value)
    plt.colorbar(label="Range (m)")
    plt.tight_layout()
    plt.show()


def barrier(x: torch.Tensor, v_to_pt):
    """障碍物避让损失函数（二次惩罚）"""
    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def is_save_iter(i):
    """判断当前迭代是否需要保存结果"""
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


time_steps = round(args.episode_length_s / args.ctl_dt)  # 每个episode的时间步数

# 主训练循环
pbar = tqdm(range(args.num_iters), ncols=80)  # 进度条
B = args.batch_size
for i in pbar:
    # 重置环境和模型状态
    env.reset()
    model.reset()

    # 初始化历史记录容器
    p_history = []  # 位置历史
    v_history = []  # 速度历史
    target_v_history = []  # 目标速度历史
    vec_to_pt_history = []  # 到最近点的向量历史
    act_diff_history = []  # 动作变化历史
    v_preds = []  # 预测速度历史
    v_net_feats = []  # 网络特征缓存
    h = None  # GRU隐藏状态

    act_lag = 0  # 动作延迟
    act_buffer = [env.act] * (act_lag + 1)  # 动作缓冲区

    # 计算初始目标速度
    target_v_raw = env.p_target - env.p

    # 偏航漂移模拟（如果启用）
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 * args.ctl_dt)  # 5度/秒的标准差
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack(
            [
                torch.cos(drift_av),
                -torch.sin(drift_av),
                zeros,
                torch.sin(drift_av),
                torch.cos(drift_av),
                zeros,
                zeros,
                zeros,
                ones,
            ],
            -1,
        ).reshape(B, 3, 3)

    # 时间步循环 (一个episode)

    for t in range(time_steps):
        # 随机控制时间间隔（模拟现实时间变化）
        ctl_dt = normalvariate(args.ctl_dt, 0.1 * args.ctl_dt)

        # 渲染深度图和光流图
        depth, _ = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())  # 计算到最近障碍物的向量

        if args.video:
            # 保存视频帧（特定迭代
            record_video(depth[5])

        # 更新目标速度（考虑偏航漂移）
        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        # 环境执行动作
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # 获取当前旋转矩阵
        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0  # 水平前向
        up[:, 2] = 1  # 垂直向上
        fwd = F.normalize(fwd, 2, -1)  # 归一化
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)  # 重建yaw-only旋转矩阵

        # 构建状态向量
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)  # 限速
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),  # 目标速度（机体坐标系）
            env.R[:, 2],  # Z轴方向
            env.margin[:, None],  # 安全距离
        ]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)  # 当前速度（机体坐标系）
        if not args.no_odom:  # 包含里程计信息
            state.insert(0, local_v)
        state = torch.cat(state, -1)  # 拼接状态向量

        # 预处理深度图
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02  # 深度转伪RGB+噪声
        x = F.max_pool2d(x[:, None], 3, 3)  # 降采样 (240x120 -> 80x40)

        # 模型前向传播
        act, values, h = model(x, state, h)  # 输出动作和隐藏状态

        # 转换动作到世界坐标系
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        # 应用推力估计误差校正
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)  # 存储到动作缓冲区
        v_net_feats.append(torch.cat([act, local_v, h], -1))  # 保存网络特征

        # 记录历史数据
        v_history.append(env.v)
        target_v_history.append(target_v)

    # --- 损失计算 ---
    p_history = torch.stack(p_history)
    # 地面亲和损失（保持一定高度）
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()

    act_buffer = torch.stack(act_buffer)
    v_history = torch.stack(v_history)

    # 速度跟踪损失（平滑L1）
    # 计算2秒内的帧数
    frames = round(2.0 / args.ctl_dt)
    # 速度跟踪损失（平滑L1）
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[frames:] - v_history_cum[:-frames]) / frames  # 2s 平均滑动
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1 : 1 - frames], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    # 速度预测损失（MSE）
    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    # 速度方向偏差损失
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)  # 速度在目标方向投影
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    # 控制平滑性损失

    jerk_history = act_buffer.diff(1, 0).mul(round(1 / args.ctl_dt))  # 加速度变化率
    snap_history = (
        F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(round(1 / args.ctl_dt) ** 2)
    )  # 加加速度变化率
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()  # 加速度大小惩罚
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()  # 急动度惩罚
    loss_d_snap = snap_history.pow(2).sum(-1).mean()  # 加急动度惩罚

    # 障碍物避让损失
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)  # 到最近障碍物距离
    distance = distance - env.margin  # 减去安全裕度
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 9 * round(1 / args.ctl_dt)).clamp_min(1)  # 障碍物接近速度
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)  # 二次障碍物损失
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()  # 碰撞损失

    # 速度大小损失
    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    # 总损失（加权求和）
    loss = (
        args.coef_v * loss_v
        + args.coef_obj_avoidance * loss_obj_avoidance
        + args.coef_bias * loss_bias
        + args.coef_d_acc * loss_d_acc
        + args.coef_d_jerk * loss_d_jerk
        + args.coef_d_snap * loss_d_snap
        + args.coef_speed * loss_speed
        + args.coef_v_pred * loss_v_pred
        + args.coef_collide * loss_collide
        + args.coef_ground_affinity * loss_ground_affinity
    )

    # 检查NaN损失
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    # 更新进度条描述
    pbar.set_description_str(f"loss: {loss:.3f}")

    # 反向传播和优化
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()  # 更新学习率

    # --- 评估与记录 ---
    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        # 判断任务成功（整个episode无碰撞）
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B  # 成功率

        # 收集指标
        smooth_dict(
            {
                "loss": loss,
                "loss_v": loss_v,
                "loss_v_pred": loss_v_pred,
                "loss_obj_avoidance": loss_obj_avoidance,
                "loss_d_acc": loss_d_acc,
                "loss_d_jerk": loss_d_jerk,
                "loss_d_snap": loss_d_snap,
                "loss_bias": loss_bias,
                "loss_speed": loss_speed,
                "loss_collide": loss_collide,
                "loss_ground_affinity": loss_ground_affinity,
                "success": _success,
                "max_speed": speed_history.max(0).values.mean(),
                "avg_speed": avg_speed.mean(),
                "ar": (success * avg_speed).mean(),
            }
        )  # 平均速度×成功率

        # # 定期保存可视化结果
        # if is_save_iter(i):
        #     # 位置历史图
        #     fig_p, ax = plt.subplots()
        #     p_history_sample = p_history[:, 4].cpu()
        #     ax.plot(p_history_sample[:, 0], label="x")
        #     ax.plot(p_history_sample[:, 1], label="y")
        #     ax.plot(p_history_sample[:, 2], label="z")
        #     ax.legend()

        #     # 速度历史图
        #     fig_v, ax = plt.subplots()
        #     v_history_sample = v_history[:, 4].cpu()
        #     ax.plot(v_history_sample[:, 0], label="x")
        #     ax.plot(v_history_sample[:, 1], label="y")
        #     ax.plot(v_history_sample[:, 2], label="z")
        #     ax.legend()

        #     # 动作历史图
        #     fig_a, ax = plt.subplots()
        #     act_buffer_sample = act_buffer[:, 4].cpu()
        #     ax.plot(act_buffer_sample[:, 0], label="x")
        #     ax.plot(act_buffer_sample[:, 1], label="y")
        #     ax.plot(act_buffer_sample[:, 2], label="z")
        #     ax.legend()

        #     # 写入TensorBoard
        #     writer.add_figure("p_history", fig_p, i + 1)
        #     writer.add_figure("v_history", fig_v, i + 1)
        #     writer.add_figure("a_reals", fig_a, i + 1)

        # 定期保存模型
        if (i + 1) % 1000 == 0:
            # -- Save PPO model
            # saved_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optim.state_dict(), "iter": i}
            # -- Save observation normalizer if used
            saved_dict = {
                "model_state_dict": model.state_dict(),
            }
            if args.video:
                video_writer.release()
                print("视频保存完成！")
            torch.save(model.state_dict(), f"{log_dir}/checkpoint{i}.pt")

        # 定期记录指标到TensorBoard
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)  # 写入平均值
            scaler_q.clear()  # 清空队列
