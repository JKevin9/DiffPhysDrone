"""Independent procedural scenes; no map files or benchmark trial inputs.

Floating primitives follow FOREST_UNIFORM_CFG: floor(scale * poles) per 8x8 tile,
uniform among active_floating_types, dimensions/radii and height from mesh_terrains.py.
Solid analytic primitives replace the tessellated reference geometry.
"""

import json
import time
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path

FLOATING_TYPES = ("box", "sphere", "ellipsoid")

TASK_PATTERNS = (
    ("far_distance_high_speed", 0.4, (130.0, 200.0), (25.0, 40.0)),
    ("middle_distance_middle_speed", 0.3, (20.0, 100.0), (6.0, 19.0)),
    ("near_distance_low_middle_speed", 0.2, (0.5, 20.0), (1.0, 19.0)),
    ("all_random", 0.1, (0.5, 200.0), (1.0, 40.0)),
)


@dataclass(frozen=True)
class SceneConfig:
    size: tuple = (256.0, 32.0)
    tile_size: float = 8.0
    poles_per_tile: tuple = (10, 10)
    pole_radius: tuple = (0.1, 0.3)
    pole_tilt_deg: tuple = (0.0, 60.0)
    roof_height: tuple = (6.0, 10.0)
    spawn_agl: tuple = (0.5, 5.5)
    wall_thickness: float = 0.3
    floating_count_scale: float = 0.8
    active_floating_types: tuple = FLOATING_TYPES
    floating_height: tuple = (0.0, 10.0)
    floating_box_size: tuple = (0.1, 0.6)
    floating_sphere_radius: tuple = (0.1, 0.3)
    floating_ellipsoid_radius: tuple = (0.1, 0.6)
    floating_ellipsoid_scale: tuple = (0.5, 1.5)
    clearance: float = 0.38  # 0.20 m maximum drone radius + 0.18 m margin
    max_sample_attempts: int = 10000

    def __post_init__(self):
        for name in (
            "pole_radius",
            "pole_tilt_deg",
            "roof_height",
            "spawn_agl",
            "floating_height",
            "floating_box_size",
            "floating_sphere_radius",
            "floating_ellipsoid_radius",
            "floating_ellipsoid_scale",
        ):
            low, high = getattr(self, name)
            if not np.isfinite((low, high)).all() or low > high:
                raise ValueError(f"Invalid {name} range")
        if len(self.size) != 2 or any(s <= 0 or s % self.tile_size for s in self.size):
            raise ValueError("Scene size must be positive multiples of tile_size")
        if self.tile_size != 8:
            raise ValueError("Density is defined per 8x8 m tile")
        if np.hypot(*self.size) <= 200 + 2 * self.clearance:
            raise ValueError("Scene must accommodate the 200 m task range")
        count = self.poles_per_tile
        if isinstance(count, int):
            object.__setattr__(self, "poles_per_tile", (count, count))
            count = self.poles_per_tile
        if len(count) != 2 or any(int(c) != c for c in count) or not 0 <= count[0] <= count[1]:
            raise ValueError("poles_per_tile must be an integer or [min, max] integer range")
        if not (6 <= self.roof_height[0] <= self.roof_height[1] <= 10):
            raise ValueError("Roof height must remain in [6, 10] m")
        if not (0.5 <= self.spawn_agl[0] <= self.spawn_agl[1] <= 5.5):
            raise ValueError("Spawn AGL must remain in [0.5, 5.5] m")
        if not 0 <= self.pole_tilt_deg[0] <= self.pole_tilt_deg[1] < 85:
            raise ValueError("Pole tilt must be in [0, 85) degrees")
        if self.pole_radius[0] <= 0 or self.clearance <= 0 or self.wall_thickness <= 0 or self.floating_count_scale < 0:
            raise ValueError("Geometry sizes/clearance must be positive and counts nonnegative")
        types = self.active_floating_types
        if isinstance(types, str):
            types = (types,)
        types = tuple(types)
        if not types or any(name not in FLOATING_TYPES for name in types) or len(set(types)) != len(types):
            raise ValueError("active_floating_types must be unique names among box, sphere, ellipsoid")
        object.__setattr__(self, "active_floating_types", types)
        if self.max_sample_attempts < 1:
            raise ValueError("max_sample_attempts must be positive")

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def metadata(self):
        return asdict(self)


def rotations(rng, n, tilt):
    # Uniform azimuth and configurable inclination, no vehicle attitude changes.
    if n == 0:
        return np.zeros((0, 3, 3))
    yaw, azimuth = rng.uniform(-np.pi, np.pi, (2, n))
    angle = np.deg2rad(rng.uniform(tilt[0], tilt[1], n))
    z = np.stack((np.sin(angle) * np.cos(azimuth), np.sin(angle) * np.sin(azimuth), np.cos(angle)), 1)
    x = np.stack((np.cos(yaw), np.sin(yaw), np.zeros(n)), 1)
    x -= z * np.sum(x * z, axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack((x, np.cross(z, x), z), 2)


def primitive(kind, center, radii, frame=None):
    frame = np.eye(3) if frame is None else frame
    radii = np.asarray(radii)
    return np.r_[kind, center, radii, frame.reshape(-1), np.linalg.norm(radii)].astype(np.float32)


def pack_primitives(kind, center, radii, frame):
    center = np.asarray(center, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    out = np.empty((len(center), 17), np.float32)
    out[:, 0] = kind
    out[:, 1:4] = center
    out[:, 4:7] = radii
    out[:, 7:16] = np.asarray(frame, dtype=np.float32).reshape(len(center), 9)
    out[:, 16] = np.linalg.norm(radii, axis=1)
    return out


def point_clearance(objects, point):
    delta = np.asarray(point) - objects[:, 1:4]
    local = np.einsum("ni,nij->nj", delta, objects[:, 7:16].reshape(-1, 3, 3))
    radii = objects[:, 4:7]
    kind = objects[:, 0]
    distance = np.linalg.norm(np.maximum(np.abs(local) - radii, 0), axis=1)
    cylinders = kind == 3
    q = local[cylinders]
    r = radii[cylinders]
    distance[cylinders] = np.hypot(
        np.maximum(np.linalg.norm(q[:, :2], axis=1) - r[:, 0], 0), np.maximum(np.abs(q[:, 2]) - r[:, 2], 0)
    )
    ellipses = kind == 2
    distance[ellipses] = np.maximum(np.linalg.norm(local[ellipses] / radii[ellipses], axis=1) - 1, 0) * radii[
        ellipses
    ].min(axis=1)
    return float(distance.min())


class ProceduralScene:
    def __init__(self, config, batch_size, device, seed):
        self.config, self.batch_size, self.device = config, batch_size, torch.device(device)
        self.rng = np.random.default_rng(seed)

    def _geometry(self):
        c, rng = self.config, self.rng
        sx, sy = c.size
        roof = float(rng.uniform(*c.roof_height))
        wall = c.wall_thickness
        walls = np.stack(
            (
                primitive(1, (sx / 2, sy / 2, -wall / 2), (sx / 2 + wall, sy / 2 + wall, wall / 2)),
                primitive(1, (sx / 2, sy / 2, roof + wall / 2), (sx / 2 + wall, sy / 2 + wall, wall / 2)),
                primitive(1, (-wall / 2, sy / 2, roof / 2), (wall / 2, sy / 2 + wall, roof / 2 + wall)),
                primitive(1, (sx + wall / 2, sy / 2, roof / 2), (wall / 2, sy / 2 + wall, roof / 2 + wall)),
                primitive(1, (sx / 2, -wall / 2, roof / 2), (sx / 2, wall / 2, roof / 2 + wall)),
                primitive(1, (sx / 2, sy + wall / 2, roof / 2), (sx / 2, wall / 2, roof / 2 + wall)),
            )
        )
        xs, ys = np.arange(0, sx, c.tile_size), np.arange(0, sy, c.tile_size)
        tile_x, tile_y = np.repeat(xs, len(ys)), np.tile(ys, len(xs))
        counts = rng.integers(c.poles_per_tile[0], c.poles_per_tile[1] + 1, len(tile_x))
        parts = [walls, self._poles(roof, tile_x, tile_y, counts), self._floating(tile_x, tile_y, counts)]
        return np.concatenate([part for part in parts if len(part)]), roof, counts.tolist()

    def _poles(self, roof, tile_x, tile_y, counts):
        tile = np.repeat(np.arange(len(counts)), counts)
        if tile.size == 0:
            return np.zeros((0, 17), np.float32)
        frame = rotations(self.rng, tile.size, self.config.pole_tilt_deg)
        radius = self.rng.uniform(*self.config.pole_radius, tile.size)
        length = (roof + 4) / frame[:, 2, 2]
        base = np.column_stack(
            (
                tile_x[tile] + self.rng.uniform(0, self.config.tile_size, tile.size),
                tile_y[tile] + self.rng.uniform(0, self.config.tile_size, tile.size),
                np.full(tile.size, -1.0),
            )
        )
        center = base + frame[:, :, 2] * (length / 2)[:, None]
        radii = np.column_stack((radius, radius, length / 2))
        return pack_primitives(3, center, radii, frame)

    def _floating(self, tile_x, tile_y, counts):
        c, rng = self.config, self.rng
        per_tile = np.fromiter((int(int(n) * c.floating_count_scale) for n in counts), np.int64, len(counts))
        tile = np.repeat(np.arange(len(counts)), per_tile)
        if tile.size == 0:
            return np.zeros((0, 17), np.float32)
        choice = rng.integers(0, len(c.active_floating_types), tile.size)
        names = np.asarray(c.active_floating_types, dtype=object)[choice]
        center = np.column_stack(
            (
                tile_x[tile] + rng.uniform(0, c.tile_size, tile.size),
                tile_y[tile] + rng.uniform(0, c.tile_size, tile.size),
                rng.uniform(*c.floating_height, tile.size),
            )
        )
        radii = np.empty((tile.size, 3))
        box, sphere, ellipsoid = names == "box", names == "sphere", names == "ellipsoid"
        if np.any(box):
            radii[box] = rng.uniform(*c.floating_box_size, (int(box.sum()), 3)) / 2
        if np.any(sphere):
            radii[sphere] = rng.uniform(*c.floating_sphere_radius, int(sphere.sum()))[:, None]
        if np.any(ellipsoid):
            count = int(ellipsoid.sum())
            radii[ellipsoid] = rng.uniform(*c.floating_ellipsoid_radius, count)[:, None] * rng.uniform(
                *c.floating_ellipsoid_scale, (count, 3)
            )
        frame = np.broadcast_to(np.eye(3), (tile.size, 3, 3)).copy()
        oriented = box | ellipsoid
        if np.any(oriented):
            frame[oriented] = rotations(rng, int(oriented.sum()), (0.0, 90.0))
        return pack_primitives(np.where(box, 1, 2), center, radii, frame)

    def _task(self, objects, roof):
        c, rng = self.config, self.rng
        index = int(rng.choice(4, p=[p[1] for p in TASK_PATTERNS]))
        name, _, distances, speeds = TASK_PATTERNS[index]
        speed = rng.uniform(*speeds)
        for attempt in range(1, c.max_sample_attempts + 1):
            start = np.r_[rng.uniform(c.clearance, np.asarray(c.size) - c.clearance), rng.uniform(*c.spawn_agl)]
            target_z = rng.uniform(*c.spawn_agl)
            distance = rng.uniform(*distances)
            dz = target_z - start[2]
            if abs(dz) > distance:
                continue
            angle = rng.uniform(-np.pi, np.pi)
            horizontal = np.sqrt(distance**2 - dz**2)
            target = start + np.array([np.cos(angle) * horizontal, np.sin(angle) * horizontal, dz])
            if np.any(target[:2] < c.clearance) or np.any(target[:2] > np.asarray(c.size) - c.clearance):
                continue
            if max(start[2], target[2]) > roof - c.clearance:
                continue
            if point_clearance(objects, start) >= c.clearance and point_clearance(objects, target) >= c.clearance:
                self.last_sample_attempts = attempt
                return start, target, speed, name
        self.last_sample_attempts = c.max_sample_attempts
        raise RuntimeError(f"Cannot sample legal {name} endpoints; reduce obstacle density or enlarge scene")

    def reset(self):
        geometries, starts, targets, speeds, records = [], [], [], [], []
        geometry_s = task_s = 0.0
        attempts = []
        object_counts = []
        for _ in range(self.batch_size):
            started = time.perf_counter()
            objects, roof, counts = self._geometry()
            geometry_s += time.perf_counter() - started
            started = time.perf_counter()
            start, target, speed, category = self._task(objects, roof)
            task_s += time.perf_counter() - started
            attempts.append(self.last_sample_attempts)
            object_counts.append(len(objects))
            geometries.append(objects)
            starts.append(start)
            targets.append(target)
            speeds.append(speed)
            records.append(dict(roof_height=roof, category=category, speed=speed, poles_per_tile=counts))
        started = time.perf_counter()
        padded = np.zeros((self.batch_size, max(len(g) for g in geometries), 17), dtype=np.float32)
        for i, geometry in enumerate(geometries):
            padded[i, : len(geometry)] = geometry
        self.objects = torch.as_tensor(padded, device=self.device)
        self.records = records
        self.start = torch.tensor(np.asarray(starts), dtype=torch.float32, device=self.device)
        self.target = torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device)
        self.speed = torch.tensor(speeds, dtype=torch.float32, device=self.device)[:, None]
        upload_s = time.perf_counter() - started
        self.timing = {
            "time_scene_geometry_s": geometry_s,
            "time_scene_task_s": task_s,
            "time_scene_upload_s": upload_s,
            "scene_task_attempts_mean": float(np.mean(attempts)),
            "scene_objects_mean": float(np.mean(object_counts)),
            "scene_objects_max": int(max(object_counts)),
        }
        return self.start, self.target, self.speed

    @torch.no_grad()
    def raycast(self, starts, directions, max_distance):
        from .cuda_backend import scene_backend

        return scene_backend().raycast(self.objects, starts.contiguous(), directions.contiguous(), max_distance)

    @torch.no_grad()
    def closest(self, points):
        from .cuda_backend import scene_backend

        return scene_backend().closest(self.objects, points.contiguous())
