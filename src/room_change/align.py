"""Gravity alignment: find the floor and rotate the scene so the floor is z = 0 with +z up."""

from pathlib import Path

import numpy as np
import open3d as o3d

from .scene import Scene, load_vggt


def find_floor(scene: Scene, n_planes: int = 8, dist: float = 0.01):
    """Return (unit normal pointing up, point on plane) of the floor.

    The phone is held roughly upright, so the cameras' mean image "down" axis
    approximates gravity. Among the dominant planes (RANSAC) the floor is the
    horizontal one that lies furthest along that down direction.
    """
    o3d.utility.random.seed(0)  # RANSAC is random; fixed seed keeps the aligned frame reproducible
    pts = np.concatenate([scene.cloud(v, max_points=200_000)[0] for v in (0, 1)])
    down = scene.extrinsic[:, 1, :3].mean(0)
    down /= np.linalg.norm(down)

    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    min_inliers = 0.03 * len(pts)
    best = None
    for _ in range(n_planes):
        plane, inl = rest.segment_plane(dist, 3, 2000)
        if len(inl) < min_inliers:
            break
        n = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        p = np.asarray(rest.points)[inl]
        if abs(n @ down) > 0.95:  # horizontal
            height = p.mean(0) @ down  # larger = lower in the room
            if best is None or height > best[0]:
                up = -n if n @ down > 0 else n
                best = (height, up, p.mean(0))
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        raise RuntimeError("No horizontal plane found for the floor")
    return best[1], best[2]


def rotation_to_z(up: np.ndarray, forward_hint: np.ndarray) -> np.ndarray:
    """Rotation whose rows are (x, y, z) axes: z = up, y = forward_hint projected onto the floor."""
    z = up / np.linalg.norm(up)
    y = forward_hint - (forward_hint @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def gravity_align(scene: Scene, scale: float = 1.0) -> np.ndarray:
    """Rotate/translate (and optionally scale) the scene so the floor is z = 0, +z up. Returns the 4x4 used."""
    up, on_floor = find_floor(scene)
    # Point +y along the first camera's viewing direction so the top-down view matches the first frame
    R = rotation_to_z(up, scene.extrinsic[0, 2, :3])
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = -(R @ on_floor)[2]  # drop the floor to z = 0, keep the x/y origin
    scene.apply_similarity(T, scale)
    return T


def scale_from_camera_height(scene: Scene, camera_height_m: float) -> float:
    """Metres per reconstruction unit, assuming the phone was held camera_height_m above the floor.

    VGGT output has no real-world scale. The thresholds downstream (depth tolerance, object
    size, obstacle height, robot radius) are in metres, so a rough scale is enough: a handheld
    phone is typically 1.2 to 1.6 m high, which keeps the error around 10%.
    The scene must already be gravity aligned (floor at z = 0).
    """
    return camera_height_m / float(np.median(scene.camera_centers()[:, 2]))


def prepare_scene(npz: Path, camera_height_m: float = 1.4) -> tuple[Scene, float]:
    """Load a VGGT reconstruction, level it (floor z = 0, +z up) and scale it to metres.

    Returns the scene and the metres-per-unit scale.
    """
    scene = load_vggt(Path(npz))
    gravity_align(scene)
    scale = scale_from_camera_height(scene, camera_height_m)
    scene.apply_similarity(np.eye(4), scale)
    return scene, scale
