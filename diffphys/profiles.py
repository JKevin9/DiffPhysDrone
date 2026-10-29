"""Single source of truth for training and benchmark policy inputs."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    control_dt: float
    benchmark_period_steps: int
    image_height: int
    image_width: int
    sensor: str
    state_dim: int = 10
    action_dim: int = 6
    hidden_dim: int = 192
    safety_margin: float = 0.18
    speed_min: float = 1.0
    speed_max: float = 40.0
    iterations: int = 50_000
    fov_x_half_tan: float = 0.82
    camera_height: int = 48
    camera_width: int = 64
    camera_angle_deg: float = 20.0
    scan_channels: int = 110
    scan_width: int = 110
    scan_period: float = 0.1
    scan_max_distance: float = 60.0
    scan_offset: tuple = (0.0, 0.0, 0.09)
    # Yunque calibration.yaml, ENU/FLU link -> COM; scan offset is link-relative.
    scan_link_to_com: tuple = (-0.0003833965, -0.0000890752, -0.020348031594647806)
    scan_hfov_deg: tuple = (-110.0, 110.0)
    scan_vfov_deg: tuple = (-80.0, 80.0)
    depth_hfov_deg: float = 180.0
    depth_vfov_deg: float = 160.0
    min_depth: float = 0.15
    max_depth: float = 40.0
    equal_sin: bool = True
    yaw_only: bool = True
    exp_scale: float = 10.0
    exp_decay: float = 1.0
    linear_scale: float = 0.1
    offset: float = 4.0
    attitude_perturbation: bool = False
    train_depth_noise: bool = True
    camera_encoded_noise_std: float = 0.02
    depth_noise_std: float = 0.04
    depth_dropout_prob: float = 0.02
    depth_block_prob: float = 0.05
    depth_noise_immunity_prob: float = 0.2

    def metadata(self):
        return asdict(self)


PROFILES = {
    "vanilla": Profile("vanilla", 1 / 15, 67, 12, 16, "camera", attitude_perturbation=True),
    "Newton-Modifed": Profile("Newton-Modifed", 1 / 50, 20, 80, 90, "lidar"),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown profile {name!r}; choose {tuple(PROFILES)}") from None
