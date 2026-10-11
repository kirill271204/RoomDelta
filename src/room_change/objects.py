"""Group changed points into object level change candidates."""

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .changes import CHANGED, PointLabels


@dataclass
class ObjectParams:
    voxel: float = 0.02        # metres, downsampling before clustering
    eps: float = 0.06          # metres, DBSCAN neighbourhood
    min_cluster_pts: int = 15  # DBSCAN core size (on voxels)
    min_voxels: int = 150      # clusters of at least this many 2 cm voxels are kept on size alone
    min_extent: float = 0.12   # metres, every object must be at least this big in its 2nd largest dimension
    min_thickness: float = 0.04  # metres, smallest dimension; thinner slivers are below the depth tolerance
                                 # (ChangeParams.tol_abs) and are surface noise, not objects
    min_height: float = 0.05   # metres above the floor; floor patches are not objects
    # Small objects seen from afar (a box on a far counter) give few points. They are kept only on strong
    # evidence: most cameras that looked at the spot saw through it. Measured on two scenes, real small
    # changes have a see-through ratio of 0.84 to 0.95, depth noise clusters mostly 0.60 to 0.79.
    small_min_voxels: int = 50
    small_min_extent: float = 0.10
    small_min_through_ratio: float = 0.8
    small_min_through: float = 3.0


@dataclass
class ChangeObject:
    kind: str                  # "removed" (only in before) or "added" (only in after)
    points: np.ndarray         # (K,3) metres
    colors: np.ndarray         # (K,3) uint8
    point_index: np.ndarray    # (K,) indices into the source PointLabels
    centroid: np.ndarray = field(init=False)
    bbox_min: np.ndarray = field(init=False)
    bbox_max: np.ndarray = field(init=False)
    support: float = 0.0       # mean number of other video views that saw through it
    through_ratio: float = 0.0 # mean share of informative views that saw through it

    def __post_init__(self):
        self.centroid = self.points.mean(0)
        self.bbox_min = np.percentile(self.points, 2, axis=0)
        self.bbox_max = np.percentile(self.points, 98, axis=0)

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def gap_to(self, other: "ChangeObject") -> float:
        """Distance between the two bounding boxes (0 if they touch or overlap)."""
        d = np.maximum(0, np.maximum(other.bbox_min - self.bbox_max, self.bbox_min - other.bbox_max))
        return float(np.linalg.norm(d))

    def merged_with(self, other: "ChangeObject") -> "ChangeObject":
        merged = ChangeObject(self.kind, np.concatenate([self.points, other.points]),
                              np.concatenate([self.colors, other.colors]),
                              np.concatenate([self.point_index, other.point_index]))
        n = len(merged.points)
        merged.support = (self.support * len(self.points) + other.support * len(other.points)) / n
        merged.through_ratio = (self.through_ratio * len(self.points) + other.through_ratio * len(other.points)) / n
        return merged


def cluster_changes(labels: PointLabels, kind: str, p: ObjectParams = ObjectParams()) -> list[ChangeObject]:
    """Cluster CHANGED points into objects and drop slivers, specks and floor patches."""
    idx = np.flatnonzero((labels.label == CHANGED) & (labels.points[:, 2] > p.min_height))
    if len(idx) == 0:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(labels.points[idx].astype(np.float64)))
    down, _, members = pcd.voxel_down_sample_and_trace(p.voxel, pcd.get_min_bound(), pcd.get_max_bound())
    vox_label = np.asarray(down.cluster_dbscan(p.eps, p.min_cluster_pts))

    objects = []
    for c in range(vox_label.max() + 1):
        vox = np.flatnonzero(vox_label == c)
        if len(vox) < p.small_min_voxels:
            continue
        pt_idx = idx[np.concatenate([np.asarray(members[v]) for v in vox])]
        obj = ChangeObject(kind, labels.points[pt_idx], labels.colors[pt_idx], pt_idx)
        thru, cons = labels.n_through[pt_idx], labels.n_consistent[pt_idx]
        obj.support = float(thru.mean())
        obj.through_ratio = float(np.mean(thru / np.maximum(thru + cons, 1)))
        dims = np.sort(obj.size)
        if dims[0] < p.min_thickness:  # sheets along edges are depth artefacts
            continue
        large = len(vox) >= p.min_voxels and dims[1] >= p.min_extent
        small = (dims[1] >= p.small_min_extent and obj.through_ratio >= p.small_min_through_ratio
                 and obj.support >= p.small_min_through)
        if large or small:
            objects.append(obj)
    return sorted(objects, key=lambda o: -len(o.points))
