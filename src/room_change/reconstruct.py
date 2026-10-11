"""Run VGGT on the frames of both recordings in one pass (one shared 3D frame).

Works on CUDA (Colab, cloud) and on Apple Silicon (MPS). On a 16 GB Mac about
15 frames per recording fit in memory; a CUDA GPU with 16 GB+ handles 30+.
The output .npz has the same layout as notebooks/vggt_colab.ipynb.
"""

import sys
from pathlib import Path

import numpy as np
import torch

VGGT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "vggt"


def _import_vggt():
    # VGGT is used from a clone, not pip: its pyproject pins numpy 1.26, which conflicts with numpy 2
    if not VGGT_DIR.exists():
        raise SystemExit(f"VGGT code not found at {VGGT_DIR}. Run: "
                         f"git clone https://github.com/facebookresearch/vggt.git {VGGT_DIR}")
    sys.path.insert(0, str(VGGT_DIR))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        return "cuda", torch.bfloat16 if bf16 else torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def run_vggt(frame_dirs: list[Path], names: list[str], frames_per_video: int, out: Path) -> None:
    VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri = _import_vggt()
    device, dtype = pick_device()

    paths, video_id = [], []
    for v, d in enumerate(frame_dirs):
        files = sorted(d.glob("*.jpg"))
        if not files:
            raise SystemExit(f"No frames in {d}")
        keep = np.linspace(0, len(files) - 1, min(frames_per_video, len(files))).astype(int)
        paths += [files[i] for i in keep]
        video_id += [v] * len(keep)
    print(f"VGGT on {device} ({dtype}), {len(paths)} frames")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    # 'pad' keeps the full portrait frame instead of cropping its top and bottom
    images = load_and_preprocess_images([str(p) for p in paths], mode="pad").to(device)
    with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])

    f32 = lambda t: t.squeeze(0).float().cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=(images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8),
        depth=f32(pred["depth"])[..., 0],
        depth_conf=f32(pred["depth_conf"]),
        world_points=f32(pred["world_points"]),
        world_points_conf=f32(pred["world_points_conf"]),
        extrinsic=f32(extrinsic),
        intrinsic=f32(intrinsic),
        video_id=np.array(video_id),
        video_names=np.array(names),
        paths=np.array([f"{p.parent.name}/{p.name}" for p in paths]),
    )
    print(f"saved {out}")


def memory_warning(n_frames: int) -> str | None:
    """Rough check that n_frames fit in memory; returns a warning or None.

    Measured on Apple Silicon (MPS, float16): about 11 GB + 0.23 GB per frame
    (10 frames: 13.0 GB, 30 frames: 17.7 GB). Attention grows faster than linearly,
    so this underestimates large runs. Only checked on Apple Silicon: CUDA memory use is
    much lower (a 15 GB T4 runs 60 frames), and on CPU there is no fixed limit.
    """
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        return None
    import os
    need = 11 + 0.23 * n_frames
    have, where = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, "unified memory"
    if need <= 0.9 * have:
        return None
    fit = max(2, int((0.9 * have - 11) / 0.23) // 2)
    return (f"{n_frames} frames need roughly {need:.0f} GB, this machine has {have:.0f} GB of {where}. "
            f"It may swap heavily or fail; --frames-per-video {fit} should fit (fewer frames can miss small objects), "
            f"or run notebooks/vggt_colab.ipynb on a GPU and pass --reconstruction.")
