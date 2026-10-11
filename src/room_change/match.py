"""Decide which removed/added pairs are the same object that moved.

Each object is described by DINOv2 features of several masked crops (background
blanked out, so the furniture or floor behind it does not count) and by its 3D size.
Appearance similarity is the best agreement between any view of one object and
any view of the other: a chair seen only from behind in the first video can still
match the same chair seen from the side in the second. Pairs are chosen with an
optimal one-to-one assignment and only accepted if the combined score is high
enough; otherwise the objects stay "removed" and "added".
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .objects import ChangeObject

_MODEL = None


def _dino():
    global _MODEL
    if _MODEL is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(device).eval()
    return _MODEL


def embed(images: list[Image.Image]) -> np.ndarray:
    """(n, d) unit length DINOv2 CLS embeddings, one per image."""
    model = _dino()
    device = next(model.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    batch = []
    for im in images:
        x = torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) / 255).permute(2, 0, 1)
        batch.append((x - mean) / std)
    with torch.no_grad():
        f = model(torch.stack(batch).to(device)).cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def size_similarity(a: ChangeObject, b: ChangeObject) -> float:
    """1 for identical sorted box dimensions, toward 0 as they differ (rotation invariant)."""
    sa, sb = np.sort(a.size), np.sort(b.size)
    return float(np.mean(np.minimum(sa, sb) / np.maximum(sa, sb)))


@dataclass
class MatchParams:
    w_appearance: float = 0.7
    min_appearance: float = 0.45   # cosine similarity of DINOv2 embeddings
    min_score: float = 0.5         # combined appearance + size score


@dataclass
class Match:
    removed: int        # index into removed objects
    added: int          # index into added objects
    appearance: float
    size: float
    score: float


def match_objects(removed: list[ChangeObject], added: list[ChangeObject],
                  emb_removed: list[np.ndarray], emb_added: list[np.ndarray], p: MatchParams = MatchParams()):
    """Return (matches, score matrix). emb_* hold one (views, d) embedding array per object."""
    if not removed or not added:
        return [], np.zeros((len(removed), len(added)))
    app = np.array([[float((er @ ea.T).max()) for ea in emb_added] for er in emb_removed])
    size = np.array([[size_similarity(r, a) for a in added] for r in removed])
    score = p.w_appearance * app + (1 - p.w_appearance) * size
    rows, cols = linear_sum_assignment(-score)
    matches = [Match(r, c, float(app[r, c]), float(size[r, c]), float(score[r, c]))
               for r, c in zip(rows, cols) if app[r, c] >= p.min_appearance and score[r, c] >= p.min_score]
    return matches, score
