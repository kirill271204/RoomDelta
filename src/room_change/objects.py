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
    min_voxels: int = 150      # drop clusters smaller than this many 2 cm voxels
    min_extent: float = 0.12   # metres, every object must be at least this big in its 2nd largest dimension
    min_thickness: float = 0.05  # metres, smallest dimension; thinner slivers are below the depth tolerance
                                 # (ChangeParams.tol_abs) and are surface noise, not objects
    min_height: float = 0.05   # metres above the floor; floor patches are not objects


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

    def __post_init__(self):
        self.centroid = self.points.mean(0)
        self.bbox_min = np.percentile(self.points, 2, axis=0)
        self.bbox_max = np.percentile(self.points, 98, axis=0)

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


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
        if len(vox) < p.min_voxels:
            continue
        pt_idx = idx[np.concatenate([np.asarray(members[v]) for v in vox])]
        obj = ChangeObject(kind, labels.points[pt_idx], labels.colors[pt_idx], pt_idx)
        dims = np.sort(obj.size)
        if dims[1] < p.min_extent or dims[0] < p.min_thickness:  # lines and sheets along edges are depth artefacts
            continue
        obj.support = float(labels.n_through[pt_idx].mean())
        objects.append(obj)
    return sorted(objects, key=lambda o: -len(o.points))
