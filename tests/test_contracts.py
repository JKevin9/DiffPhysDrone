"""Executable policy, geometry and AIRNet contract regressions."""

import importlib.util
import numpy as np
import torch
from pathlib import Path

import pytest
from diffphys.checkpoints import load_checkpoint, save_checkpoint
from main_cuda import DepthVideo, build_parser
from diffphys.models import build_model
from diffphys.perception import (
    LidarHistory,
    camera_directions,
    decode_action,
    lidar_directions,
    policy_state,
    project_points,
    transform_depth,
    yaw_rotation,
)
from diffphys.profiles import PROFILES, get_profile
from diffphys.runtime import PolicyRuntime
from diffphys.scenes import TASK_PATTERNS, ProceduralScene, SceneConfig, point_clearance, primitive


@pytest.mark.parametrize("name", PROFILES)
def test_checkpoint_roundtrip_rejects_other_profile_and_nonfinite(tmp_path, name):
    spec = get_profile(name)
    model = build_model(spec)
    x = torch.randn(2, 1, spec.image_height, spec.image_width)
    state = torch.randn(2, 10)
    out, _, h = model(x, state)
    assert out.shape == (2, 6) and h.shape == (2, 192)
    if name != "vanilla":
        assert model.stem[:6](torch.zeros(2, 1, 32, 32)).shape == (2, 128, 8, 8)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, spec, 4, run_config={"seed": 42})
    restored, payload = load_checkpoint(path, spec)
    assert payload["iteration"] == 4
    torch.testing.assert_close(restored(x, state)[0], out)
    other = next(key for key in PROFILES if key != name)
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, other)
    payload["model_state_dict"]["fc.weight"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_checkpoint(path, spec)


def test_depth_video_stops_after_its_frame_budget():
    class Writer:
        def __init__(self):
            self.frames = []
            self.released = 0

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.released += 1

    args = build_parser().parse_args(["--profile", "vanilla"])
    assert args.video_seconds == 60.0 and not args.video
    assert int(args.video_seconds / get_profile("vanilla").control_dt + 1e-9) == 900
    assert int(60.0 / get_profile("Newton-Modifed").control_dt + 1e-9) == 3000
    writer = Writer()
    video = DepthVideo(writer, 2)
    video.write("a")
    video.write("b")
    video.write("c")
    video.close()
    assert writer.frames == ["a", "b"] and writer.released == 1 and not video.active


def test_raw_legacy_checkpoint_is_not_silently_accepted(tmp_path):
    path = tmp_path / "raw.pt"
    torch.save(build_model("vanilla").state_dict(), path)
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path, "vanilla")


def test_zero_distance_state_and_interleaved_world_action():
    rotation = torch.tensor([[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    p = torch.zeros(1, 3)
    velocity = torch.tensor([[0.0, 2, 0]])
    state, frame, desired = policy_state(p, velocity, rotation, p, torch.tensor([40.0]))
    assert torch.isfinite(state).all() and desired.count_nonzero() == 0
    torch.testing.assert_close(state[:, :3], torch.tensor([[2.0, 0, 0]]))
    torch.testing.assert_close(state[:, -1], torch.tensor([0.18]))
    acceleration, prediction = decode_action(torch.tensor([[4.0, 1, 8, 2, 12, 3]]), frame)
    torch.testing.assert_close(acceleration, torch.tensor([[-6.0, 3, 9]]))
    torch.testing.assert_close(prediction, torch.tensor([[-2.0, 1, 3]]))


@pytest.mark.parametrize("name", PROFILES)
def test_physics_clock_holds_actions_and_resets_only_ended_environment(name):
    spec = get_profile(name)

    class Constant(torch.nn.Module):
        def forward(self, image, state, hidden):
            return torch.tensor([[2.0, 0, 0, 0, 0, 0]]).expand(len(state), -1), None, hidden + 1

    runtime = PolicyRuntime(Constant(), spec, 2, "cpu")

    def image():
        return torch.zeros(2, 1, spec.image_height, spec.image_width)

    p = torch.zeros(2, 3)
    r = torch.eye(3).repeat(2, 1, 1)
    speed = torch.ones(2)
    ticks = []
    for step in range(2 * spec.benchmark_period_steps + 1):
        acceleration, updated = runtime.update(step, image, p, p, r, p + 1, speed)
        if updated:
            ticks.append(step)
        if step < spec.benchmark_period_steps:
            assert acceleration.count_nonzero() == 0 and runtime.hidden.count_nonzero() == 0
    assert ticks == [0, spec.benchmark_period_steps, 2 * spec.benchmark_period_steps]
    assert torch.all(runtime.acceleration[:, 0] == 2)
    survivor = runtime.hidden[1].clone()
    runtime.reset(torch.tensor([0]))
    assert runtime.hidden[0].count_nonzero() == 0
    torch.testing.assert_close(runtime.hidden[1], survivor)
    runtime.update(ticks[-1] + 1, image, p, p, r, p + 1, speed)
    assert runtime.acceleration[0].count_nonzero() == 0
    assert runtime.acceleration[1, 0] == 2


def test_original_camera_projection_and_pooling():
    spec = get_profile("vanilla")
    directions = camera_directions(spec, "cpu").reshape(48, 64, 3)
    assert torch.all(directions[..., 0] == 1)
    assert directions[0, 0, 1] > 0 and directions[0, 0, 2] > 0
    depth = torch.linspace(0.1, 100, 48 * 64).reshape(1, 48, 64)
    reference = torch.nn.functional.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
    torch.testing.assert_close(transform_depth(depth, spec), reference)


def aerial_module(relative):
    root = Path(__file__).resolve().parents[2] / "aerial_stack/python_repos/aerial_lab/aerial/aerial"
    path = root / relative
    if not path.is_file():
        pytest.skip(f"AIRNet source not available: {path}")
    spec = importlib.util.spec_from_file_location("parity_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_airnet_projection_transform_no_hits_and_current_pose_parity():
    reference = aerial_module("utils/converter.py")
    transform = aerial_module("mdp/observations/lidar_transform.py")
    spec = get_profile("Newton-Modifed")
    torch.manual_seed(14)
    points = torch.randn(3, 12100, 3) * 20
    points[:, :10] = float("inf")
    points[:, 10:20] = 0
    converter = reference.PC2DepthImgConverter(80, 90, 40.0, 0.15, 40.0, 160.0, 180.0, True)
    depth = project_points(points, spec)
    torch.testing.assert_close(depth, converter.convert(points), rtol=0, atol=0)
    expected = transform._apply_depth_transform(
        depth,
        "reward_like",
        min_depth=0.15,
        max_depth=40.0,
        reward_like_params=dict(exp_scale=10.0, exp_decay=1.0, linear_scale=0.1, offset=4.0),
    )
    torch.testing.assert_close(transform_depth(depth, spec)[:, 0], expected, rtol=0, atol=0)
    no_hits = torch.full((1, 12100, 3), float("inf"))
    assert torch.all(project_points(no_hits, spec) == 40)
    pattern = reference.build_lidar_pattern_directions_f64(
        channels=110, width=110, vertical_fov_range_deg=(-80, 80), horizontal_fov_range_deg=(-110, 110)
    )
    np.testing.assert_allclose(lidar_directions(spec, "cpu").numpy(), pattern, atol=4e-7)


def test_cached_world_scan_reprojects_without_rescanning():
    spec = get_profile("Newton-Modifed")
    history = LidarHistory(spec, "cpu")

    class Scene:
        calls = 0

        def raycast(self, starts, directions, maximum):
            self.calls += 1
            assert maximum == 60
            torch.testing.assert_close(starts[:, 2], torch.tensor([2.09 - spec.scan_link_to_com[2]]))
            return torch.full(directions.shape[:2], 5.0)

    scene = Scene()
    p = torch.tensor([[0.0, 0, 2.0]])
    r = torch.eye(3)[None]
    first = history.observe(scene, p, r, 0.0)
    cached = history.points_w.clone()
    moved = p + torch.tensor([[0.2, 0, 0]])
    rotated = torch.tensor([[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    for time in (0.02, 0.04, 0.06, 0.08):
        value = history.observe(scene, moved, rotated, time)
        expected = project_points((cached - moved[:, None]) @ yaw_rotation(rotated), spec)
        torch.testing.assert_close(value, expected)
    assert scene.calls == 1 and not torch.equal(first, value)
    history.observe(scene, p, r, 0.1)
    assert scene.calls == 2
    history.reset()
    history.observe(scene, p, r, 0.0)
    assert scene.calls == 3


@pytest.mark.parametrize("counts", [(0, 0), (2, 4)])
def test_scene_ground_roof_walls_and_endpoint_clearance(counts):
    config = SceneConfig(poles_per_tile=counts)
    scene = ProceduralScene(config, 3, "cpu", 42)
    scene.reset()
    twin = ProceduralScene(config, 3, "cpu", 42)
    twin.reset()
    torch.testing.assert_close(scene.objects, twin.objects)
    for i, record in enumerate(scene.records):
        objects = scene.objects[i].numpy()
        objects = objects[objects[:, 0] != 0]
        roof = record["roof_height"]
        assert 6 <= roof <= 10
        assert objects[0, 3] + objects[0, 6] == 0
        assert np.isclose(objects[1, 3] - objects[1, 6], roof)
        assert len(record["poles_per_tile"]) == 128
        assert all(counts[0] <= n <= counts[1] for n in record["poles_per_tile"])
        expected = 6 + sum(n + int(n * 0.8) for n in record["poles_per_tile"])
        assert len(objects) == expected
        for point in (scene.start[i].numpy(), scene.target[i].numpy()):
            assert 0.5 <= point[2] <= 5.5
            assert point_clearance(objects, point) >= config.clearance - 1e-5
            assert np.all(point[:2] >= config.clearance)
            assert np.all(point[:2] <= np.asarray(config.size) - config.clearance)
    assert scene.timing["scene_task_attempts_mean"] >= 1
    assert scene.timing["time_scene_geometry_s"] >= 0
    assert scene.timing["time_scene_task_s"] >= 0
    assert scene.timing["scene_objects_max"] >= 6


def test_active_floating_types_omit_ellipsoids():
    config = SceneConfig(poles_per_tile=2, active_floating_types=["box", "sphere"])
    scene = ProceduralScene(config, 1, "cpu", 1)
    scene.reset()
    objects = scene.objects[0].numpy()
    spheres = objects[objects[:, 0] == 2]
    assert len(spheres) > 0
    eye = np.eye(3, dtype=np.float32).reshape(-1)
    for obj in spheres:
        assert np.allclose(obj[4:7], obj[4])
        assert np.allclose(obj[7:16], eye)
    with pytest.raises(ValueError):
        SceneConfig(active_floating_types=["ellipsoid", "ellipsoid"])


def test_task_distribution_spans_full_speed_range_without_fixed_trials():
    scene = ProceduralScene(SceneConfig(poles_per_tile=0), 1, "cpu", 9)
    geometry, roof, _ = scene._geometry()
    samples = [scene._task(geometry, roof) for _ in range(2000)]
    speeds = [item[2] for item in samples]
    assert 1 <= min(speeds) < 1.2 and 39.8 < max(speeds) <= 40
    for name, prob, distances, interval in TASK_PATTERNS:
        selected = [item for item in samples if item[3] == name]
        assert abs(len(selected) / len(samples) - prob) < 0.04
        for start, goal, speed, _ in selected:
            assert distances[0] - 1e-5 <= np.linalg.norm(goal - start) <= distances[1] + 1e-5
            assert interval[0] <= speed <= interval[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_geometry_ground_roof_walls_and_tilted_obstacle():
    from diffphys.cuda_backend import scene_backend

    scene = ProceduralScene(SceneConfig(poles_per_tile=0), 1, "cuda", 3)
    scene.reset()
    p = torch.tensor([[128.0, 16.0, 2.0]], device="cuda")
    directions = torch.tensor([[[0.0, 0, -1], [0, 0, 1], [0, 1, 0], [0, -1, 0]]], device="cuda")
    depth = scene.raycast(p, directions, 100.0)
    expected = torch.tensor([[2.0, scene.records[0]["roof_height"] - 2, 16.0, 16.0]], device="cuda")
    torch.testing.assert_close(depth, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scene.closest(p[:, None])[0, 0], torch.tensor([128.0, 16, 0], device="cuda"))
    frame = np.array([[1, 0, 0], [0, 2**-0.5, -(2**-0.5)], [0, 2**-0.5, 2**-0.5]])
    objects = torch.tensor(primitive(3, (0, 0, 0), (1, 1, 5), frame)[None, None], device="cuda")
    origin = torch.tensor([[3.0, 0, 0]], device="cuda")
    ray = torch.tensor([[[-1.0, 0, 0]]], device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = scene_backend().raycast(objects, origin, ray, 10.0)
    stream.synchronize()
    torch.testing.assert_close(out, torch.tensor([[2.0]], device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_same_scene_pose_scan_matches_airnet_raycaster_core():
    import math
    import trimesh

    import warp as wp
    from aerial_core.sensing.lidar import (
        RayCasterData,
        initialize_ray_caster_data,
        initialize_ray_caster_state,
        update_ray_caster_data,
    )
    from aerial_core.sensing.lidar.ray_caster_core import quat_apply

    spec = get_profile("Newton-Modifed")
    reference = aerial_module("utils/converter.py")
    pattern_module = aerial_module("sensors/ray_caster/patterns/patterns.py")
    from types import SimpleNamespace

    starts, directions = pattern_module.lidar_pattern(
        SimpleNamespace(
            channels=110, width=110, horizontal_fov_range=(-110.0, 110.0), vertical_fov_range=(-80.0, 80.0)
        ),
        "cuda",
    )
    torch.testing.assert_close(directions, lidar_directions(spec, "cuda"), atol=0, rtol=0)
    scene = ProceduralScene(SceneConfig(poles_per_tile=0), 1, "cuda", 4)
    scene.reset()
    # Both engines see exactly the same six boxes, with identical poses.
    meshes = []
    for obj in scene.objects[0].cpu().numpy():
        transform = np.eye(4)
        transform[:3, :3] = obj[7:16].reshape(3, 3)
        transform[:3, 3] = obj[1:4]
        meshes.append(trimesh.creation.box(extents=2 * obj[4:7], transform=transform))
    mesh = trimesh.util.concatenate(meshes)
    wp_mesh = wp.Mesh(
        points=wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(np.asarray(mesh.faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device="cuda:0"),
    )
    # Nonzero yaw and roll distinguish full-body COM offset from yaw-only casting.
    yaw, roll = 0.27, 0.31
    quat = torch.tensor(
        [
            [
                math.cos(yaw / 2) * math.cos(roll / 2),
                math.cos(yaw / 2) * math.sin(roll / 2),
                math.sin(yaw / 2) * math.sin(roll / 2),
                math.sin(yaw / 2) * math.cos(roll / 2),
            ]
        ],
        device="cuda",
    )
    rotation = quat_apply(quat[:, None], torch.eye(3, device="cuda")[None]).transpose(1, 2)
    position = torch.tensor([[128.0, 16.0, 2.0]], device="cuda")
    link_position = position - torch.matmul(rotation, torch.tensor(spec.scan_link_to_com, device="cuda"))
    data = RayCasterData()
    initialize_ray_caster_data(data, 1, 12100, "cuda")
    state = initialize_ray_caster_state(1, 12100, "cuda")
    with wp.ScopedStream(wp.stream_from_torch("cuda:0")):
        update_ray_caster_data(
            data=data,
            state=state,
            env_ids=None,
            num_instances=1,
            pos_w=link_position,
            quat_w=quat,
            ray_starts=(starts + torch.tensor(spec.scan_offset, device="cuda"))[None],
            ray_directions=directions[None],
            static_mesh=wp_mesh,
            dynamic_mesh=None,
            max_distance=60.0,
            ray_alignment="yaw",
            enable_motion_distortion=False,
            pattern_num_beams=0,
            use_fused=False,
        )
    history = LidarHistory(spec, "cuda")
    history.observe(scene, position, rotation, 0.0)
    finite = torch.isfinite(history.points_w).all(-1)
    assert torch.equal(finite, torch.isfinite(data.ray_hits_w).all(-1))
    torch.testing.assert_close(history.points_w[finite], data.ray_hits_w[finite], atol=3e-5, rtol=1e-6)
    converter = reference.PC2DepthImgConverter(80, 90, 40.0, 0.15, 40.0, 160.0, 180.0, True)
    for step in range(5):
        current_position = position + torch.tensor([[step * 0.015, step * 0.01, 0.0]], device="cuda")
        current_yaw = yaw + step * 0.01
        c, s = math.cos(current_yaw), math.sin(current_yaw)
        current_rotation = torch.tensor([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], device="cuda")
        ours = history.observe(scene, current_position, current_rotation, step * 0.02)
        expected = converter.convert((data.ray_hits_w - current_position[:, None]) @ yaw_rotation(current_rotation))
        torch.testing.assert_close(ours, expected, atol=5e-5, rtol=1e-5)
    assert history.scan_count == 1


def test_airnet_training_noise_matches_reference_and_preserves_immune_rows():
    import ast
    from dataclasses import replace
    from types import SimpleNamespace

    from diffphys.perception import add_lidar_noise

    root = Path(__file__).resolve().parents[2] / "aerial_stack/python_repos/aerial_lab/aerial/aerial"
    path = root / "mdp/observations/basic_observation.py"
    if not path.is_file():
        pytest.skip("AIRNet source not available")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LidarDepthImgWrapper")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_depth_noise")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    spec = replace(get_profile("Newton-Modifed"), depth_block_prob=1.0)
    reference = SimpleNamespace(
        gaussian_noise_std=spec.depth_noise_std,
        min_depth=spec.min_depth,
        max_depth=spec.max_depth,
        invalid_depth=spec.max_depth,
        block_prob=spec.depth_block_prob,
        loss_prob=spec.depth_dropout_prob,
    )
    depth = torch.full((3, 80, 90), 7.0)
    immune = torch.tensor([False, True, False])
    torch.manual_seed(34)
    ours = add_lidar_noise(depth, spec, immune)
    torch.manual_seed(34)
    expected = namespace["_apply_depth_noise"](reference, depth, immune)
    torch.testing.assert_close(ours, expected, rtol=0, atol=0)
    torch.testing.assert_close(ours[1], depth[1], rtol=0, atol=0)
    assert (ours[0] == 40).any()


def test_tensorboard_tags_are_grouped_by_meaning():
    from main_cuda import TENSORBOARD_TAGS, tensorboard_tag

    groups = {tag.split("/", 1)[0] for tag in TENSORBOARD_TAGS.values()}
    assert groups == {"loss", "flight", "task", "scene", "time"}
    assert len(set(TENSORBOARD_TAGS.values())) == len(TENSORBOARD_TAGS)
    assert tensorboard_tag("loss") == "loss/total"
    assert tensorboard_tag("loss_collide") == "loss/safety/collide"
    assert tensorboard_tag("time_scene_upload_s") == "time/scene/upload"
    assert tensorboard_tag("sampled_speed_max") == "task/sampled_speed_max"
    with pytest.raises(KeyError, match="ungrouped"):
        tensorboard_tag("not_a_metric")
