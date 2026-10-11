"""Visibility aware 3D change detection between two posed RGB-D recordings.

Every 3D point from one recording is projected into every camera of the other
recording and compared with the depth that camera measured at that pixel:

    measured depth ~= point depth   -> the surface is still there   (consistent)
    measured depth  > point depth   -> camera saw THROUGH the point  (free space violation)
    measured depth  < point depth   -> something is in front         (occluded, no evidence)
    point outside the image / low confidence pixel                   (not observed, no evidence)

A point only becomes "changed" when the other recording actually looked at that
spot and saw empty space there. Points that the other recording never observed,
or only saw blocked by something else, are labelled "unknown" instead of
"changed". This is what separates real changes from viewpoint, occlusion and
incomplete coverage.

Before that, each point must be confirmed by other frames of its OWN recording
(multi view support). Things that are only in one or two frames, such as a hand
in front of the lens or depth noise at object edges, are dropped as "transient".
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

from .scene import Scene

UNCHANGED, CHANGED, UNKNOWN, TRANSIENT = 0, 1, 2, 3
LABEL_NAMES = {UNCHANGED: "unchanged", CHANGED: "changed", UNKNOWN: "unknown", TRANSIENT: "transient"}


@dataclass
class ChangeParams:
    conf_pct: float = 50          # confidence percentile for building each recording's own point cloud
    evidence_conf_pct: float = 0  # depth used as evidence keeps all pixels: plain floors and walls get
                                  # minimum VGGT confidence, but their depth is still good enough to
                                  # show that space is empty (multi view voting handles the noise)
    tol_abs: float = 0.06         # metres: depth agreement tolerance ...
    tol_rel: float = 0.04         # ... plus this fraction of the depth
    edge_px: int = 2              # depth window half size, guards against depth edges
    min_self_support: int = 2     # other frames of the same video that must confirm a point
    min_views: int = 2            # informative views of the other video needed to decide
    changed_ratio: float = 0.6    # fraction of informative views that must see through it
    unchanged_ratio: float = 0.3  # below this fraction of see-through views it is unchanged


@dataclass
class PointLabels:
    points: np.ndarray       # (M,3) world, metres
    colors: np.ndarray       # (M,3) uint8
    label: np.ndarray        # (M,) one of UNCHANGED/CHANGED/UNKNOWN/TRANSIENT
    n_consistent: np.ndarray # (M,) views of the other video that saw the surface there
    n_through: np.ndarray    # (M,) views of the other video that saw past it


class DepthBank:
    """Depth maps of a set of frames, ready for fast projection tests."""

    def __init__(self, scene: Scene, frames: np.ndarray, p: ChangeParams, conf_pct: float):
        self.frames = frames
        self.K = scene.intrinsic[frames]
        self.E = scene.extrinsic[frames]
        conf_mask = scene.confidence_mask(int(scene.video_id[frames[0]]), conf_pct)[frames]
        valid = binary_erosion(scene.valid, iterations=p.edge_px + 1)  # skip the padding border
        self.ok = conf_mask & valid[None]
        size = 2 * p.edge_px + 1
        d = scene.depth[frames]
        # Near/far envelopes: a point is only "seen through" if it is in front of even the
        # nearest surface in a small window, so depth edges do not create fake changes.
        self.d_min = np.stack([minimum_filter(x, size=size) for x in d])
        self.d_max = np.stack([maximum_filter(x, size=size) for x in d])
        self.h, self.w = d.shape[1:]

    def compare(self, pts: np.ndarray, p: ChangeParams, skip_frame: np.ndarray | None = None):
        """Count, per point, frames where the surface is consistent and frames that see through it."""
        n_cons = np.zeros(len(pts), np.int32)
        n_thru = np.zeros(len(pts), np.int32)
        for k, f in enumerate(self.frames):
            cam = pts @ self.E[k, :, :3].T + self.E[k, :, 3]
            z = cam[:, 2]
            front = z > 0.05
            uv = np.full((len(pts), 2), -1, np.int64)
            uv[front] = np.round((cam[front, :2] / z[front, None]) @ self.K[k, :2, :2].T + self.K[k, :2, 2]).astype(np.int64)
            inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < self.w) & (uv[:, 1] >= 0) & (uv[:, 1] < self.h)
            if skip_frame is not None:
                inside &= skip_frame != f
            idx = np.flatnonzero(inside)
            u, v = uv[idx, 0], uv[idx, 1]
            seen = self.ok[k, v, u]
            idx, u, v = idx[seen], u[seen], v[seen]
            tol = p.tol_abs + p.tol_rel * z[idx]
            dmin, dmax = self.d_min[k, v, u], self.d_max[k, v, u]
            n_thru[idx[dmin > z[idx] + tol]] += 1
            n_cons[idx[(dmin <= z[idx] + tol) & (dmax >= z[idx] - tol)]] += 1
        return n_cons, n_thru


def detect(scene: Scene, source: int, p: ChangeParams = ChangeParams(), max_points: int = 300_000) -> PointLabels:
    """Label every point of recording `source` against the other recording."""
    target = 1 - source
    pts, cols, src_frame = scene.cloud(source, p.conf_pct, max_points)

    own = DepthBank(scene, scene.frames_of(source), p, p.evidence_conf_pct)
    self_cons, _ = own.compare(pts, p, skip_frame=src_frame)

    other = DepthBank(scene, scene.frames_of(target), p, p.evidence_conf_pct)
    n_cons, n_thru = other.compare(pts, p)

    n_obs = n_cons + n_thru
    ratio = np.divide(n_thru, n_obs, out=np.zeros(len(pts)), where=n_obs > 0)
    label = np.full(len(pts), UNKNOWN, np.int8)
    decided = n_obs >= p.min_views
    label[decided & (ratio >= p.changed_ratio) & (n_thru >= p.min_views)] = CHANGED
    label[decided & (ratio <= p.unchanged_ratio)] = UNCHANGED
    label[self_cons < p.min_self_support] = TRANSIENT
    return PointLabels(pts, cols, label, n_cons, n_thru)


def summary(labels: PointLabels) -> dict[str, float]:
    n = len(labels.label)
    return {LABEL_NAMES[k]: round(100 * float((labels.label == k).sum()) / n, 1) for k in LABEL_NAMES}
