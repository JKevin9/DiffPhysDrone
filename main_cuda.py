"""Train either versioned DiffPhys profile on independent procedural scenes."""

import argparse
import json
import numpy as np
import random
import time
import torch
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffphys.checkpoints import load_checkpoint, save_checkpoint
from diffphys.git_snapshot import capture_git_snapshot
from diffphys.models import build_model
from diffphys.perception import decode_action, policy_state
from diffphys.profiles import PROFILES, get_profile
from diffphys.scenes import TASK_PATTERNS, SceneConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--num_iters", "--num-iters", type=int, default=50000)
    parser.add_argument("--episode_length_s", "--episode-length-s", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--exp_name", default="procedural")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--video", action="store_true", help="Record consumed model depth inputs")
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=60.0,
        help="Maximum depth-video duration in seconds; used only with --video",
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA-synchronize and record reset, perception, nearest, forward, dynamics, backward",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--airmode_extra_force_coef", type=float, default=0.0)
    for name, default in dict(
        v=1.0,
        speed=0.0,
        v_pred=0.0,
        collide=7.5,
        obj_avoidance=3.0,
        d_acc=0.01,
        d_jerk=0.001,
        d_snap=0.0,
        ground_affinity=0.0,
        bias=0.0,
    ).items():
        parser.add_argument(f"--coef_{name}", type=float, default=default)
    return parser


class DepthVideo:
    """Write at most max_frames control steps, then release the file."""

    def __init__(self, writer, max_frames):
        self._writer = writer
        self.remaining = max_frames

    @property
    def active(self):
        return self.remaining > 0

    def write(self, frame):
        if self.remaining <= 0:
            return
        self._writer.write(frame)
        self.remaining -= 1
        if self.remaining == 0:
            self.close()

    def close(self):
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()


@contextmanager
def depth_video(path, profile, enabled, max_seconds):
    if not enabled:
        yield None
        return
    import cv2

    max_frames = int(max_seconds / profile.control_dt + 1e-9)
    if max_frames < 1:
        raise ValueError(f"--video-seconds {max_seconds} is shorter than one control step")
    video = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1 / profile.control_dt,
        (profile.image_width, profile.image_height),
        isColor=False,
    )
    if not video.isOpened():
        raise RuntimeError(f"Cannot open depth video: {path}")
    recorder = DepthVideo(video, max_frames)
    try:
        yield recorder
    finally:
        recorder.close()


class StageTimer:
    """Accumulate CUDA-synchronized stage times. Disabled timers do not synchronize."""

    def __init__(self, device, enabled):
        self.device = torch.device(device)
        self.enabled = enabled
        self.totals = {}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)

    def reset(self):
        self.totals = {}

    def metrics(self):
        values = {f"time_{name}_s": seconds for name, seconds in self.totals.items()}
        values["time_total_s"] = float(sum(self.totals.values()))
        return values


def rollout_loss(env, model, args, video=None, timer=None):
    spec = env.profile
    dt = spec.control_dt
    steps = round(args.episode_length_s / dt)
    timer = timer or StageTimer(env.device, False)
    positions, velocities, desireds, nearest, predictions = [], [], [], [], []
    hidden = None
    # Preserve the existing one-policy-step act_buffer delay.
    actions = [env.act]
    for _ in range(steps):
        with timer.section("obs"):
            image = env.observation(training=True)
        if video is not None and video.active:
            with timer.section("video"):
                if spec.sensor == "camera":
                    low, high = 3 / 24 - 0.6, 3 / 0.3 - 0.6
                else:
                    low, high = (
                        spec.exp_scale * np.exp(-spec.exp_decay * d) - spec.linear_scale * d + spec.offset
                        for d in (spec.max_depth, spec.min_depth)
                    )
                frame = (image[0, 0].detach().clamp(low, high) - low) / (high - low)
                video.write((frame * 255).byte().cpu().numpy())
        with timer.section("forward"):
            state, frame, desired = policy_state(env.p, env.v, env.R, env.p_target, env.max_speed, spec.safety_margin)
            positions.append(env.p)
            velocities.append(env.v)
            desireds.append(desired)
            action, _, hidden = model(image, state.detach(), hidden)
            acceleration, predicted = decode_action(action, frame)
        with timer.section("nearest"):
            nearest.append(env.find_vec_to_nearest_pt())
        predictions.append(predicted)
        acceleration = (acceleration - env.g_std) * env.thr_est_error[:, None] + env.g_std
        previous = actions[-1]
        actions.append(acceleration)
        with timer.section("dynamics"):
            env.run(previous, dt, env.p_target - env.p.detach())
    with timer.section("loss"):
        position, velocity, desired = map(torch.stack, (positions, velocities, desireds))
        predictions, action = torch.stack(predictions), torch.stack(actions)
        frames = min(round(2 / dt), steps - 1)
        cumulative = velocity.cumsum(0)
        average = (cumulative[frames:] - cumulative[:-frames]) / frames
        reference = desired[1 : 1 - frames] if frames > 1 else desired[1:]
        delta = (average - reference).norm(dim=-1)
        direction = F.normalize(desired, dim=-1)
        forward_speed = (velocity * direction).sum(-1)
        distance = torch.stack(nearest).norm(dim=-1) - env.margin
        with torch.no_grad():
            approach = (-distance.diff(dim=1) * 9 / dt).clamp_min(1)
        losses = {
            "v": F.smooth_l1_loss(delta, torch.zeros_like(delta)),
            "v_pred": F.mse_loss(predictions, velocity.detach()),
            "bias": F.mse_loss(velocity, forward_speed[..., None] * direction) * 3,
            "d_acc": action.square().sum(-1).mean(),
            "d_jerk": (action.diff(dim=0) / dt).square().sum(-1).mean(),
            "d_snap": (F.normalize(action - env.g_std, dim=-1).diff(dim=0).diff(dim=0) / dt**2).square().sum(-1).mean(),
            "obj_avoidance": (approach * (1 - distance[:, 1:]).relu().square()).mean(),
            "collide": (F.softplus(-32 * distance[:, 1:]) * approach).mean(),
            "speed": F.smooth_l1_loss(forward_speed, desired.norm(dim=-1)),
            "ground_affinity": position[..., 2].relu().square().mean(),
        }
        loss = sum(getattr(args, f"coef_{name}") * value for name, value in losses.items())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        metrics = {f"loss_{name}": value.detach().item() for name, value in losses.items()}
        metrics.update(
            loss=loss.detach().item(),
            collision_free_fraction=(distance.flatten(0, 1) > 0).all(0).float().mean().item(),
            avg_speed=velocity.detach().norm(dim=-1).mean().item(),
            sampled_speed_min=env.max_speed.min().item(),
            sampled_speed_max=env.max_speed.max().item(),
        )
    return loss, metrics, steps


LOG_WIDTH = 80
LOG_PAD = 35
# TensorBoard groups tags on "/". metrics.jsonl keeps the flat keys.
TENSORBOARD_TAGS = {
    "loss": "loss/total",
    "loss_v": "loss/tracking/v",
    "loss_v_pred": "loss/tracking/v_pred",
    "loss_bias": "loss/tracking/bias",
    "loss_speed": "loss/tracking/speed",
    "loss_d_acc": "loss/smooth/acc",
    "loss_d_jerk": "loss/smooth/jerk",
    "loss_d_snap": "loss/smooth/snap",
    "loss_obj_avoidance": "loss/safety/obj_avoidance",
    "loss_collide": "loss/safety/collide",
    "loss_ground_affinity": "loss/safety/ground_affinity",
    "collision_free_fraction": "flight/collision_free",
    "avg_speed": "flight/avg_speed",
    "sampled_speed_min": "task/sampled_speed_min",
    "sampled_speed_max": "task/sampled_speed_max",
    "scene_task_attempts_mean": "scene/task_attempts_mean",
    "scene_objects_mean": "scene/objects_mean",
    "scene_objects_max": "scene/objects_max",
    "time_reset_s": "time/rollout/reset",
    "time_obs_s": "time/rollout/obs",
    "time_nearest_s": "time/rollout/nearest",
    "time_forward_s": "time/rollout/forward",
    "time_dynamics_s": "time/rollout/dynamics",
    "time_loss_s": "time/rollout/loss",
    "time_backward_s": "time/learn/backward",
    "time_optim_s": "time/learn/optim",
    "time_scene_geometry_s": "time/scene/geometry",
    "time_scene_task_s": "time/scene/task",
    "time_scene_upload_s": "time/scene/upload",
    "time_video_s": "time/io/video",
    "time_log_s": "time/io/log",
    "time_checkpoint_s": "time/io/checkpoint",
    "time_total_s": "time/total",
}
LOSS_ROWS = (
    ("loss_v", "Velocity loss"),
    ("loss_v_pred", "Velocity prediction loss"),
    ("loss_bias", "Bias loss"),
    ("loss_d_acc", "Acceleration loss"),
    ("loss_d_jerk", "Jerk loss"),
    ("loss_d_snap", "Snap loss"),
    ("loss_obj_avoidance", "Obstacle avoidance loss"),
    ("loss_collide", "Collision loss"),
    ("loss_speed", "Speed loss"),
    ("loss_ground_affinity", "Ground affinity loss"),
)


def tensorboard_tag(key):
    try:
        return TENSORBOARD_TAGS[key]
    except KeyError:
        raise KeyError(f"ungrouped tensorboard metric: {key}") from None


def write_tensorboard(writer, metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(tensorboard_tag(key), value, step)


def _row(label, value):
    return f"{f'{label}:':>{LOG_PAD}} {value}"


def _title(text):
    return f"\033[1m{text.center(LOG_WIDTH)}\033[0m"


def format_banner(spec, args, log_dir, first_iteration, steps, git_state):
    rows = [
        _row("Profile", spec.name),
        _row("Control dt", f"{spec.control_dt * 1000:.1f} ms"),
        _row("Batch", args.batch_size),
        _row("Steps/iter", steps),
        _row("Iterations", f"{first_iteration + 1} → {args.num_iters}"),
        _row("Timing", "on" if args.timing else "off"),
        _row("Seed", args.seed),
        _row("Log dir", log_dir),
        _row("Commit", f"{git_state['commit'][:12]} ({'dirty' if git_state['dirty'] else 'clean'})"),
    ]
    return "\n".join(("#" * LOG_WIDTH, _title(f" Training {spec.name} "), "", *rows, ""))


def format_status(iteration, total, metrics, steps, batch_size, timing_enabled, iteration_time, total_time, eta):
    fps = steps * batch_size / max(iteration_time, 1e-9)
    if timing_enabled:
        collection = sum(
            metrics.get(name, 0.0)
            for name in (
                "time_reset_s",
                "time_obs_s",
                "time_nearest_s",
                "time_forward_s",
                "time_dynamics_s",
                "time_loss_s",
                "time_video_s",
            )
        )
        learning = metrics.get("time_backward_s", 0.0) + metrics.get("time_optim_s", 0.0)
        computation = f"{fps:.0f} steps/s (collection: {collection:.3f}s, learning {learning:.3f}s)"
    else:
        computation = f"{fps:.0f} steps/s"
    rows = [
        _row("Computation", computation),
        _row("Mean loss", f"{metrics['loss']:.4f}"),
    ]
    rows.extend(_row(label, f"{metrics[key]:.4f}") for key, label in LOSS_ROWS if key in metrics)
    rows.extend(
        (
            _row("Collision free", f"{100 * metrics['collision_free_fraction']:.4f}"),
            _row("Mean speed", f"{metrics['avg_speed']:.2f} m/s"),
            _row(
                "Sampled speed",
                f"{metrics['sampled_speed_min']:.1f}-{metrics['sampled_speed_max']:.1f} m/s",
            ),
            _row("Scene geometry", f"{metrics['time_scene_geometry_s']:.3f}s"),
            _row("Scene task", f"{metrics['time_scene_task_s']:.3f}s"),
            _row("Scene upload", f"{metrics['time_scene_upload_s']:.3f}s"),
            _row("Sample attempts", f"{metrics['scene_task_attempts_mean']:.1f}"),
            _row("Scene objects", f"{metrics['scene_objects_mean']:.0f}/{metrics['scene_objects_max']}"),
        )
    )
    if timing_enabled:

        def step_ms(name):
            return 1000.0 * metrics[f"time_{name}_s"] / steps

        rows.extend(
            _row(label, f"{step_ms(name):.2f} ms/step")
            for name, label in (
                ("obs", "Observation"),
                ("nearest", "Nearest point"),
                ("forward", "Forward"),
                ("dynamics", "Dynamics"),
            )
        )
    if "time_video_s" in metrics:
        rows.append(_row("Video", f"{metrics['time_video_s']:.3f}s"))
    if "time_checkpoint_s" in metrics:
        rows.append(_row("Checkpoint", f"{metrics['time_checkpoint_s']:.3f}s"))
    footer = [
        _row("Iteration time", f"{iteration_time:.2f}s"),
        _row("Total time", f"{total_time:.2f}s"),
        _row("ETA", f"{eta:.1f}s"),
    ]
    return "\n".join(
        (
            "#" * LOG_WIDTH,
            _title(f" Learning iteration {iteration}/{total} "),
            "",
            *rows,
            "-" * LOG_WIDTH,
            *footer,
        )
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = get_profile(args.profile)
    if (
        args.batch_size < 1
        or args.num_iters < 1
        or args.save_every < 1
        or round(args.episode_length_s / spec.control_dt) < 3
    ):
        parser.error("positive batch/iteration/save intervals and at least three episode steps required")
    if args.video_seconds <= 0 or (args.video and int(args.video_seconds / spec.control_dt + 1e-9) < 1):
        parser.error("--video-seconds must cover at least one control step")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        parser.error("Differentiable training requires a CUDA device")
    scene_config = SceneConfig.from_file(args.scene_config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from torch.utils.tensorboard import SummaryWriter

    from env_cuda import Env

    env = Env(
        args.batch_size,
        grad_decay=args.grad_decay,
        device=args.device,
        profile=spec,
        scene_config=scene_config,
        seed=args.seed,
        airmode_extra_force_coef=args.airmode_extra_force_coef,
    )
    payload = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, spec, args.device)
    else:
        model = build_model(spec).to(args.device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.num_iters, args.lr * 0.01)
    first_iteration = 0
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        first_iteration = int(payload["iteration"])
        if first_iteration >= args.num_iters:
            parser.error("--num_iters must exceed the checkpoint's completed iteration")
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + spec.name
    if args.run_name:
        name += "_" + args.run_name
    log_dir = (args.log_root / args.exp_name / name).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    git_state = capture_git_snapshot(log_dir)
    run_config = dict(
        profile=spec.metadata(),
        scene=scene_config.metadata(),
        task_patterns=TASK_PATTERNS,
        seed=args.seed,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        training_maps="independent procedural geometry",
        policy_action_delay_steps=1,
        resume_semantics="optimizer/scheduler restored; sampling restarts from the explicit seed",
        git=git_state,
    )
    (log_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    episode_steps = round(args.episode_length_s / spec.control_dt)
    print(format_banner(spec, args, log_dir, first_iteration, episode_steps, git_state), flush=True)
    timer = StageTimer(args.device, args.timing)
    train_started = time.perf_counter()
    with (
        SummaryWriter(log_dir=str(log_dir)) as writer,
        (log_dir / "metrics.jsonl").open("w") as metrics_file,
        depth_video(log_dir / "depth.mp4", spec, args.video, args.video_seconds) as video,
    ):
        for i in range(first_iteration, args.num_iters):
            iter_started = time.perf_counter()
            timer.reset()
            with timer.section("reset"):
                env.reset()
            loss, metrics, steps = rollout_loss(env, model, args, video=video, timer=timer)
            with timer.section("backward"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite policy gradient")
            with timer.section("optim"):
                optimizer.step()
                scheduler.step()
            metrics.update(env.scene.timing)
            if args.timing:
                metrics.update(timer.metrics())
            log_started = time.perf_counter()
            write_tensorboard(writer, metrics, i + 1)
            metrics["time_log_s"] = time.perf_counter() - log_started
            if (i + 1) % args.save_every == 0 or i + 1 == args.num_iters:
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    log_dir / f"checkpoint_{i + 1:06d}.pt",
                    model,
                    spec,
                    i + 1,
                    run_config=run_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                metrics["time_checkpoint_s"] = time.perf_counter() - checkpoint_started
            if args.timing:
                metrics["time_total_s"] += metrics["time_log_s"] + metrics.get("time_checkpoint_s", 0.0)
                late = {
                    "time_log_s": metrics["time_log_s"],
                    "time_total_s": metrics["time_total_s"],
                }
                if "time_checkpoint_s" in metrics:
                    late["time_checkpoint_s"] = metrics["time_checkpoint_s"]
                write_tensorboard(writer, late, i + 1)
            metrics_file.write(json.dumps(dict(iteration=i + 1, **metrics)) + "\n")
            metrics_file.flush()
            completed = i + 1 - first_iteration
            elapsed = time.perf_counter() - train_started
            print(
                format_status(
                    i + 1,
                    args.num_iters,
                    metrics,
                    steps,
                    args.batch_size,
                    args.timing,
                    time.perf_counter() - iter_started,
                    elapsed,
                    elapsed / completed * (args.num_iters - (i + 1)),
                ),
                flush=True,
            )
    return log_dir


if __name__ == "__main__":
    main()
