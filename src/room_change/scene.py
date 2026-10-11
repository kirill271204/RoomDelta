"""Load a joint VGGT reconstruction and turn it into point clouds and posed depth maps."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class Scene:
    """Posed RGB-D frames from both recordings, in one shared world frame.

    Convention: extrinsic is world-to-camera (OpenCV: x right, y down, z forward).
    """

    images: np.ndarray       # (N,H,W,3) uint8
    depth: np.ndarray        # (N,H,W) float32, camera z
    conf: np.ndarray         # (N,H,W) float32, VGGT depth confidence (>= 1)
    extrinsic: np.ndarray    # (N,3,4)
    intrinsic: np.ndarray    # (N,3,3)
    video_id: np.ndarray     # (N,) 0 = before, 1 = after
    video_names: list[str]
    valid: np.ndarray        # (H,W) bool, pixels that are real image (not VGGT padding)
    paths: list[str]         # (N,) source frame file per frame, relative to the frames folder

    @property
    def n_frames(self) -> int:
        return len(self.images)

    def frames_of(self, video: int) -> np.ndarray:
        return np.flatnonzero(self.video_id == video)

    def camera_centers(self) -> np.ndarray:
        R, t = self.extrinsic[:, :, :3], self.extrinsic[:, :, 3]
        return -np.einsum("nji,nj->ni", R, t)

    def apply_similarity(self, T: np.ndarray, scale: float) -> None:
        """Re-express the scene in a new world frame: x_new = scale * (R x + t), T = [R|t] (4x4)."""
        R, t = T[:3, :3], T[:3, 3]
        E = self.extrinsic
        # x_cam = Ec R^T (x_new/scale - t) + ec  ->  cameras see the same pixels, depths scale
        Rc = E[:, :, :3] @ R.T
        tc = scale * (E[:, :, 3] - np.einsum("nij,j->ni", Rc, t))
        self.extrinsic = np.concatenate([Rc, tc[:, :, None]], axis=2).astype(np.float32)
        self.depth = (self.depth * scale).astype(np.float32)

    def unproject(self, i: int) -> np.ndarray:
        """World points (H,W,3) for frame i."""
        h, w = self.depth.shape[1:]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([u, v, np.ones_like(u)], -1).astype(np.float32)
        cam = (pix @ np.linalg.inv(self.intrinsic[i]).T) * self.depth[i][..., None]
        R, t = self.extrinsic[i, :, :3], self.extrinsic[i, :, 3]
        return (cam - t) @ R

    def confidence_mask(self, video: int, pct: float) -> np.ndarray:
        """(N,H,W) mask: valid pixels of `video` frames above the given confidence percentile.

        pct = 0 keeps every valid pixel. VGGT confidence is 1 + exp(x) and sits exactly at its
        minimum of 1.0 on textureless surfaces (plain walls, floor), about 60% of pixels here,
        so any percentile up to ~60 means "above the minimum" and keeps only the top ~40%.
        """
        idx = self.frames_of(video)
        mask = np.zeros(self.conf.shape, bool)
        if pct <= 0:
            mask[idx] = self.valid
            return mask
        thresh = np.percentile(self.conf[idx][:, self.valid], pct)
        mask[idx] = self.valid & (self.conf[idx] > thresh)
        return mask

    def cloud(self, video: int, conf_pct: float = 50, max_points: int | None = None, seed: int = 0):
        """Fused (points, colors, frame index) for one recording, with flying pixels removed."""
        mask = self.confidence_mask(video, conf_pct)
        pts, cols, src = [], [], []
        for i in self.frames_of(video):
            pts.append(self.unproject(i)[mask[i]])
            cols.append(self.images[i][mask[i]])
            src.append(np.full(mask[i].sum(), i))
        pts, cols, src = map(np.concatenate, (pts, cols, src))
        if max_points and len(pts) > max_points:
            sel = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
            pts, cols, src = pts[sel], cols[sel], src[sel]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        _, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pts[keep], cols[keep], src[keep]


def load_vggt(npz: Path) -> Scene:
    d = np.load(npz)
    images = d["images"]
    # VGGT 'pad' mode fills borders with pure white; a row/column that is white in every frame is padding
    not_white = (images < 255).any(axis=(0, 3))
    valid = not_white.any(axis=1)[:, None] & not_white.any(axis=0)[None, :]
    return Scene(
        images=images,
        depth=d["depth"],
        conf=d["depth_conf"],
        extrinsic=d["extrinsic"],
        intrinsic=d["intrinsic"],
        video_id=d["video_id"],
        video_names=[str(n) for n in d["video_names"]],
        valid=valid,
        paths=[str(x) for x in d["paths"]],
    )
