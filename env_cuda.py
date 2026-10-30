import math
import random
import time
import torch
import torch.nn.functional as F

from diffphys.cuda_backend import dynamics_backend
from diffphys.perception import LidarHistory, add_lidar_noise, camera_directions, transform_depth
from diffphys.profiles import get_profile
from diffphys.scenes import ProceduralScene, SceneConfig

quadsim_cuda = dynamics_backend()


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode
    ):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode
        )
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R,
            dg,
            z_drag_coef,
            drag_2,
            pitch_ctl_delay,
            v,
            v_wind,
            act_next,
            d_act_next,
            d_p_next,
            d_v_next,
            d_a_next,
            ctx.grad_decay,
            ctx.ctl_dt,
        )
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None


run = RunFunction.apply


class Env:
    def __init__(
        self,
        batch_size,
        width=64,
        height=48,
        grad_decay=0.4,
        device="cuda",
        *,
        profile="vanilla",
        scene_config=None,
        seed=42,
        airmode_extra_force_coef=0.0,
    ):
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        self.device = torch.device(device)
        self.batch_size, self.width, self.height = batch_size, width, height
        self.grad_decay = grad_decay
        self.airmode_extra_force_coef = airmode_extra_force_coef
        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
        self.v_wind_w = torch.tensor([1.0, 1.0, 0.2], device=device)
        self.sub_div = torch.linspace(0, self.profile.control_dt, 10, device=device).reshape(-1, 1, 1)
        self.scene = ProceduralScene(scene_config or SceneConfig(), batch_size, device, seed)
        self.lidar = LidarHistory(self.profile, device) if self.profile.sensor == "lidar" else None
        self.reset()

    def reset(self):
        B, device, spec = self.batch_size, self.device, self.profile
        self.p, self.p_target, self.max_speed = self.scene.reset()
        self.sim_time = 0.0
        self.n_drones_per_group = 1
        self.drone_radius = random.uniform(0.15, 0.20)
        self.margin = torch.full((B,), spec.safety_margin, device=device)
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01
        self._fov_x_half_tan = spec.fov_x_half_tan
        if spec.attitude_perturbation:
            self._fov_x_half_tan *= 0.95 + 0.1 * random.random()
        camera_angle = torch.full((B,), spec.camera_angle_deg, device=device)
        if spec.attitude_perturbation:
            camera_angle += torch.randn(B, device=device)
        angle = camera_angle * math.pi / 180
        z, o = torch.zeros_like(angle), torch.ones_like(angle)
        self.R_cam = torch.stack((angle.cos(), z, -angle.sin(), z, o, z, angle.sin(), z, angle.cos()), -1).reshape(
            B, 3, 3
        )
        self.camera_rays = camera_directions(spec, device, self._fov_x_half_tan)
        # Identified on 2026-10-26 against aerial-core Yunque acceleration/yaw.
        # Preserve the independently fitted dynamics centers and randomization.
        self.pitch_ctl_delay = 12.32 + 1.232 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay_inv = 4.33 + 0.433 * torch.randn((B, 1), device=device)
        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1 if spec.attitude_perturbation else torch.zeros_like(self.v)
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2
        forward = F.normalize(self.p_target - self.p, dim=-1)
        if spec.attitude_perturbation:
            forward = forward + torch.randn_like(forward) * 0.2
        self.R = quadsim_cuda.update_state_vec(
            torch.zeros((B, 3, 3), device=device), self.act, forward, torch.zeros_like(self.yaw_ctl_delay_inv), 5
        )
        self.R_old, self.p_old = self.R.clone(), self.p
        # Project Yunque body drag plus 4-rotor H-force onto per-axis drag.
        # drag_2 is [quadratic, linear]; z_drag_coef scales the body-up part.
        self.drag_2 = torch.tensor([0.0041, 0.243], device=device).repeat(B, 1)
        self.z_drag_coef = torch.full((B, 1), 0.5, device=device)
        if self.lidar is not None:
            self.lidar.reset()
            self.depth_noise_immune = torch.rand(B, device=device) < spec.depth_noise_immunity_prob

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (
            forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]
        ) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack(
            [
                self_forward_vec,
                self_left_vec,
                self_up_vec,
            ],
            -1,
        )

    @torch.no_grad()
    def render(self, ctl_dt=None):
        if self.lidar is not None:
            return self.lidar.observe(self.scene, self.p, self.R, self.sim_time), None
        directions = self.camera_rays[None] @ (self.R @ self.R_cam).transpose(1, 2)
        depth = self.scene.raycast(self.p, directions, 100.0)
        return depth.reshape(self.batch_size, self.profile.camera_height, self.profile.camera_width), None

    def observation(self, training=True):
        depth, _ = self.render()
        noise = training and self.profile.train_depth_noise
        if noise and self.lidar is not None:
            depth = add_lidar_noise(depth, self.profile, self.depth_noise_immune)
        return transform_depth(depth, self.profile, noise=noise and self.profile.sensor == "camera")

    def find_vec_to_nearest_pt(self):
        points = self.p + self.v * self.sub_div
        nearest = self.scene.closest(points.transpose(0, 1).contiguous()).transpose(0, 1)
        return nearest - points

    def run(self, act_pred, ctl_dt=1 / 50, v_pred=None):
        self.sim_time += ctl_dt
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R,
            self.dg,
            self.z_drag_coef,
            self.drag_2,
            self.pitch_ctl_delay,
            act_pred,
            self.act,
            self.p,
            self.v,
            self.v_wind,
            self.a,
            self.grad_decay,
            ctl_dt,
            self.airmode_extra_force_coef,
        )
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay_inv * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def _run(self, act_pred, ctl_dt=1 / 50, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay**ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay**ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay_inv * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
