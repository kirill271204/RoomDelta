"""Find the frames that see a 3D object best and cut it out of the full resolution frames."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

from .objects import ChangeObject
from .scene import Scene


@dataclass
class Crop:
    frame: int           # scene frame index
    image: Image.Image   # crop from the original (full resolution) frame
    visible: int         # object points visible in this frame
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in original frame pixels
    mask: Image.Image    # "L" mask, same size as image: 255 where the object's 3D points project
    context: Image.Image # the whole frame with the crop box drawn, to show where the object is

    def masked(self, fill=(124, 116, 104)) -> Image.Image:
        """The crop with everything except the object replaced by a flat colour (ImageNet mean)."""
        return Image.composite(self.image, Image.new("RGB", self.image.size, fill), self.mask)


def _to_original(scene: Scene, frames_dir: Path, i: int):
    """Load frame i at full resolution and return it with the VGGT-pixel -> original-pixel mapping."""
    img = Image.open(frames_dir / scene.paths[i]).convert("RGB")
    cols = np.flatnonzero(scene.valid.any(0))
    rows = np.flatnonzero(scene.valid.any(1))
    # VGGT 'pad' mode resized the frame to the valid block, then padded it; undo both
    sx = img.width / len(cols)
    sy = img.height / len(rows)
    return img, (lambda u, v: ((u - cols[0]) * sx, (v - rows[0]) * sy))


def best_crops(scene: Scene, obj: ChangeObject, video: int, frames_dir: Path, k: int = 3,
               tol: float = 0.08, pad: float = 0.15, sample: int = 20000, seed: int = 0) -> list[Crop]:
    """Top-k crops of `obj` from frames of `video`, ranked by how many of its points are visible.

    A point counts as visible in a frame only if the frame's depth at that pixel agrees with the
    point's depth, so frames where the object is hidden behind something else are not chosen.
    """
    rng = np.random.default_rng(seed)
    pts = obj.points if len(obj.points) <= sample else obj.points[rng.choice(len(obj.points), sample, replace=False)]
    h, w = scene.depth.shape[1:]
    scored = []
    for i in scene.frames_of(video):
        cam = pts @ scene.extrinsic[i, :, :3].T + scene.extrinsic[i, :, 3]
        z = cam[:, 2]
        front = z > 0.1
        uv = (cam[front, :2] / z[front, None]) @ scene.intrinsic[i, :2, :2].T + scene.intrinsic[i, :2, 2]
        u, v = np.round(uv).astype(int).T
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, zf = u[inside], v[inside], z[front][inside]
        vis = scene.valid[v, u] & (np.abs(scene.depth[i, v, u] - zf) < tol + 0.04 * zf)
        if vis.sum() >= 50:
            scored.append((int(vis.sum()), i, u[vis], v[vis]))

    crops = []
    for visible, i, u, v in sorted(scored, key=lambda s: -s[0])[:k]:
        img, to_orig = _to_original(scene, frames_dir, i)
        x0, y0 = to_orig(np.percentile(u, 1), np.percentile(v, 1))
        x1, y1 = to_orig(np.percentile(u, 99), np.percentile(v, 99))
        px, py = pad * (x1 - x0), pad * (y1 - y0)
        box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(img.width, int(x1 + px)), min(img.height, int(y1 + py)))
        # Object mask at VGGT resolution: projected points, closed and slightly grown to a solid region
        m = np.zeros((h, w), bool)
        m[v, u] = True
        m = binary_dilation(binary_fill_holes(binary_closing(m, iterations=3)), iterations=2)
        full = Image.fromarray((m[scene.valid.any(1)][:, scene.valid.any(0)] * 255).astype(np.uint8)).resize(img.size)
        context = img.copy()
        ImageDraw.Draw(context).rectangle(box, outline=(255, 0, 255), width=6)
        crops.append(Crop(i, img.crop(box), visible, box, full.crop(box), context))
    return crops
