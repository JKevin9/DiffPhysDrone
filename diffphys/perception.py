"""AIRNet splatting parity and original DiffPhys camera preprocessing."""

import math
import torch
from torch.nn import functional as F


def yaw_rotation(rotation):
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(-1, 3, 3)


def policy_state(position, velocity, rotation, target, speed, margin=0.18):
    frame = yaw_rotation(rotation)
    delta = target - position
    distance = delta.norm(dim=-1, keepdim=True)
    desired = delta / distance.clamp_min(1e-8) * torch.minimum(distance, speed.reshape(-1, 1))
    local_v = torch.bmm(velocity[:, None], frame).squeeze(1)
    local_target = torch.bmm(desired[:, None], frame).squeeze(1)
    # Original Newton contract: world-up expressed in body coordinates (R row 2).
    up_body = rotation[:, 2, :]
    state = torch.cat((local_v, local_target, up_body, torch.full_like(speed.reshape(-1, 1), margin)), -1)
    return state, frame, desired


def decode_action(action, frame):
    # Interleaved [target_x, predicted_x, target_y, predicted_y, ...].
    target_local, predicted_local = action.reshape(-1, 3, 2).unbind(-1)
    target = torch.bmm(frame, target_local[:, :, None]).squeeze(-1)
    predicted = torch.bmm(frame, predicted_local[:, :, None]).squeeze(-1)
    return target - predicted, predicted


def camera_directions(profile, device, fov=None):
    h, w = profile.camera_height, profile.camera_width
    fov = profile.fov_x_half_tan if fov is None else fov
    u = (2 * (torch.arange(h, device=device) + 0.5) / h - 1) * fov * h / w - 1e-5
    v = (2 * (torch.arange(w, device=device) + 0.5) / w - 1) * fov - 1e-5
    u, v = torch.meshgrid(u, v, indexing="ij")
    # Unnormalized rays preserve camera-plane Z depth used by the original kernel.
    return torch.stack((torch.ones_like(u), -v, -u), -1).reshape(-1, 3)


def lidar_directions(profile, device):
    v = torch.linspace(profile.scan_vfov_deg[1], profile.scan_vfov_deg[0], profile.scan_channels, device=device)
    h = torch.linspace(*profile.scan_hfov_deg, profile.scan_width, device=device)
    v, h = torch.meshgrid(torch.deg2rad(v), torch.deg2rad(h), indexing="ij")
    return torch.stack((v.cos() * h.cos(), v.cos() * h.sin(), v.sin()), -1).reshape(-1, 3)


def project_points(points, profile):
    """Match aerial.utils.converter.PC2DepthImgConverter, including clipping order."""
    b, n, _ = points.shape
    valid = torch.isfinite(points).all(-1)
    flat = points.reshape(-1, 3)[valid.reshape(-1)]
    batch = torch.arange(b, device=points.device)[:, None].expand(b, n)[valid]
    x, y, z = flat.unbind(-1)
    radius = flat.norm(dim=-1).clamp(profile.min_depth, profile.max_depth)
    hfov, vfov = math.radians(profile.depth_hfov_deg), math.radians(profile.depth_vfov_deg)
    u = (torch.atan2(y, x) + hfov / 2) / hfov * profile.image_width
    v = (z / (radius + 1e-6) + math.sin(vfov / 2)) / (2 * math.sin(vfov / 2)) * profile.image_height
    mask = (u >= 0) & (u < profile.image_width) & (v >= 0) & (v < profile.image_height)
    indices = (
        batch[mask] * profile.image_height * profile.image_width + v[mask].long() * profile.image_width + u[mask].long()
    )
    depth = torch.full((b * profile.image_height * profile.image_width,), float("inf"), device=points.device)
    depth.scatter_reduce_(0, indices, radius[mask], reduce="amin", include_self=False)
    depth[torch.isinf(depth)] = profile.max_depth
    return depth.reshape(b, profile.image_height, profile.image_width).flip((1, 2))


def add_lidar_noise(depth, profile, immune_mask):
    """Match AIRNet post-projection Gaussian/block/dropout noise in meters."""
    output = depth.clone()
    active = ~immune_mask
    if not active.any():
        return output
    noisy = output[active].clone()
    noisy = noisy + torch.randn_like(noisy) * profile.depth_noise_std
    noisy.clamp_(profile.min_depth, profile.max_depth)
    batch, height, width = noisy.shape
    blocked = torch.rand(batch, device=depth.device) < profile.depth_block_prob
    for index in blocked.nonzero().flatten():
        row = int(torch.randint(0, height - 1, (1,), device=depth.device).item())
        col = int(torch.randint(0, width - 1, (1,), device=depth.device).item())
        end_row = int(torch.randint(row + 1, height, (1,), device=depth.device).item())
        end_col = int(torch.randint(col + 1, width, (1,), device=depth.device).item())
        noisy[int(index), row:end_row, col:end_col] = profile.max_depth
    noisy[torch.rand_like(noisy) < profile.depth_dropout_prob] = profile.max_depth
    output[active] = noisy
    return output


def transform_depth(depth, profile, noise=False):
    if profile.sensor == "camera":
        value = 3 / depth.clamp(0.3, 24) - 0.6
        if noise:
            value = value + torch.randn_like(value) * profile.camera_encoded_noise_std
        return F.max_pool2d(value[:, None], 4, 4)
    depth = depth.clamp(profile.min_depth, profile.max_depth)
    return (profile.exp_scale * torch.exp(-profile.exp_decay * depth) - profile.linear_scale * depth + profile.offset)[
        :, None
    ]


class LidarHistory:
    """10 Hz world hits, reprojected from current COM/yaw on every policy step."""

    def __init__(self, profile, device):
        self.profile = profile
        self.directions = lidar_directions(profile, device)
        self.reset()

    def reset(self):
        self.points_w = None
        self.next_scan_time = 0.0
        self.scan_count = 0

    @torch.no_grad()
    def observe(self, scene, position, rotation, time):
        frame = yaw_rotation(rotation)
        if self.points_w is None or time + 1e-8 >= self.next_scan_time:
            offset = torch.tensor(self.profile.scan_offset, device=position.device)
            link_to_com = torch.tensor(self.profile.scan_link_to_com, device=position.device)
            link_position = position - torch.matmul(rotation, link_to_com)
            starts = link_position + torch.matmul(frame, offset)
            directions = self.directions[None] @ frame.transpose(1, 2)
            depth = scene.raycast(starts, directions, self.profile.scan_max_distance)
            self.points_w = starts[:, None] + directions * depth[:, :, None]
            self.points_w[~torch.isfinite(depth)] = float("inf")
            self.scan_count += 1
            self.next_scan_time = (math.floor((time + 1e-8) / self.profile.scan_period) + 1) * self.profile.scan_period
        local = (self.points_w - position[:, None]) @ frame
        return project_points(local, self.profile)
