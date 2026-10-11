"""Top-down occupancy maps and path planning: what the changes mean for a robot.

Each recording gives a three-state grid: FREE (floor seen, nothing above it),
OCCUPIED (points between the floor clearance and the robot's height) or UNKNOWN
(never observed). The robot's map after the second recording is its old map with
every cell the second recording observed overwritten, which is how a robot
updates a stored map. A* then plans on both maps with obstacles inflated by the
robot radius, so the effect of the changes on a concrete route is visible.
"""

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass
class MapParams:
    res: float = 0.05            # metres per cell
    floor_max_z: float = 0.04    # points below this are floor
    obstacle_z: tuple = (0.10, 1.5)  # points in this height band block the robot; below 10 cm is
                                     # rug / floor depth noise (VGGT floor error is a few cm) and step-over height
    min_hits: int = 3            # points per cell to trust a floor / obstacle observation
    min_obstacle_cells: int = 4  # isolated obstacle specks smaller than this are depth noise
    robot_radius: float = 0.2    # metres, obstacles are inflated by this
    unknown_cost: float = 4.0    # A* cost multiplier for crossing unobserved cells


@dataclass
class Grid:
    origin: np.ndarray  # (2,) world x/y of cell (0, 0) corner
    res: float
    cells: np.ndarray   # (H, W) uint8, row = y, col = x

    def to_cell(self, xy) -> tuple[int, int]:
        c = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        return int(c[1]), int(c[0])

    def to_world(self, rc) -> np.ndarray:
        rc = np.asarray(rc, float)
        return self.origin + (rc[..., ::-1] + 0.5) * self.res

    @property
    def extent(self):
        h, w = self.cells.shape
        x0, y0 = self.origin
        return (x0, x0 + w * self.res, y0, y0 + h * self.res)


def grid_bounds(clouds: list[np.ndarray], res: float, margin: float = 0.2):
    pts = np.concatenate([c[:, :2] for c in clouds])
    lo = np.percentile(pts, 0.5, axis=0) - margin
    hi = np.percentile(pts, 99.5, axis=0) + margin
    shape = tuple(np.ceil((hi - lo)[::-1] / res).astype(int))
    return lo, shape


def occupancy(points: np.ndarray, origin: np.ndarray, shape, p: MapParams = MapParams(),
              floor_points: np.ndarray | None = None) -> Grid:
    """Three-state grid from one recording's points (metres, floor at z = 0).

    floor_points, if given, is a denser cloud used only for floor evidence: VGGT gives plain floor
    low confidence, so a confidence filtered cloud sees little of it, while obstacles should still
    come from confident points only.
    """
    def hits(mask, pts=points):
        c = np.floor((pts[mask, :2] - origin) / p.res).astype(int)
        ok = (c[:, 0] >= 0) & (c[:, 0] < shape[1]) & (c[:, 1] >= 0) & (c[:, 1] < shape[0])
        counts = np.zeros(shape, np.int32)
        np.add.at(counts, (c[ok, 1], c[ok, 0]), 1)
        return counts >= p.min_hits

    z = points[:, 2]
    fp = points if floor_points is None else floor_points
    floor = hits(np.abs(fp[:, 2]) < p.floor_max_z, fp)
    floor = binary_closing(floor, iterations=2)  # floor points are sparse; close small gaps
    occupied = hits((z > p.obstacle_z[0]) & (z < p.obstacle_z[1]))
    comp, _ = label(occupied, structure=np.ones((3, 3)))
    sizes = np.bincount(comp.ravel())
    occupied &= sizes[comp] >= p.min_obstacle_cells
    cells = np.full(shape, UNKNOWN, np.uint8)
    cells[floor] = FREE
    cells[occupied] = OCCUPIED
    return Grid(origin, p.res, cells)


def update(prior: Grid, observation: Grid) -> Grid:
    """Robot map update: cells the new recording observed replace the prior; the rest is kept."""
    cells = prior.cells.copy()
    seen = observation.cells != UNKNOWN
    cells[seen] = observation.cells[seen]
    return Grid(prior.origin, prior.res, cells)


def cost_map(grid: Grid, p: MapParams = MapParams()) -> np.ndarray:
    """Per-cell traversal cost; inf where the robot (a disc of robot_radius) would collide."""
    clearance = distance_transform_edt(grid.cells != OCCUPIED) * grid.res
    cost = np.where(grid.cells == UNKNOWN, p.unknown_cost, 1.0)
    cost[clearance < p.robot_radius] = np.inf
    return cost


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """8-connected A* on a cost grid. Returns (list of (row, col), total cost) or (None, inf)."""
    h, w = cost.shape
    if not np.isfinite(cost[start]) or not np.isfinite(cost[goal]):
        return None, np.inf
    steps = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    heur = lambda rc: ((rc[0] - goal[0]) ** 2 + (rc[1] - goal[1]) ** 2) ** 0.5
    g = {start: 0.0}
    parent = {start: None}
    heap = [(heur(start), start)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], g[goal]
        for dr, dc, d in steps:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w) or not np.isfinite(cost[nxt]):
                continue
            ng = g[cur] + d * 0.5 * (cost[cur] + cost[nxt])
            if ng < g.get(nxt, np.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(heap, (ng + heur(nxt), nxt))
    return None, np.inf


def snap_to_free(cost: np.ndarray, cell: tuple[int, int], max_cells: int):
    """Nearest cell the robot can occupy (finite cost) within max_cells, or None."""
    if np.isfinite(cost[cell]):
        return cell
    r0, c0 = cell
    rr, cc = np.mgrid[max(0, r0 - max_cells):r0 + max_cells + 1, max(0, c0 - max_cells):c0 + max_cells + 1]
    rr, cc = rr.ravel(), cc.ravel()
    ok = (rr < cost.shape[0]) & (cc < cost.shape[1])
    rr, cc = rr[ok], cc[ok]
    ok = np.isfinite(cost[rr, cc]) & ((rr - r0) ** 2 + (cc - c0) ** 2 <= max_cells ** 2)
    if not ok.any():
        return None
    d = (rr[ok] - r0) ** 2 + (cc[ok] - c0) ** 2
    i = np.argmin(d)
    return int(rr[ok][i]), int(cc[ok][i])


def reachable(cost: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Cells the robot can drive to from start (8-connected through finite cost cells)."""
    ok = np.isfinite(cost)
    lab, _ = label(ok, structure=np.ones((3, 3)))
    if not ok[start]:
        return np.zeros_like(ok)
    return lab == lab[start]


def path_length(grid: Grid, path) -> float:
    xy = grid.to_world(np.array(path))
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
