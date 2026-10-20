"""Render original DeepFluoro HDF5 data using its saved pelvis pose."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--specimen", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--mode", choices=["linear", "exp"], default="linear")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import h5py
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from pytorch3d_drr import AnatomyCT, Camera, Scene, CTRenderer
    with h5py.File(args.data, "r") as f:
        s = f[args.specimen]
        k = f["proj-params/intrinsic"][:]
        extrinsic = f["proj-params/extrinsic"][:]
        rows, cols = [int(f[f"proj-params/{x}"][()]) for x in ("num-rows", "num-cols")]
        sy, sx = [float(f[f"proj-params/{x}"][()]) for x in ("pixel-row-spacing", "pixel-col-spacing")]
        if rows != cols or not np.isclose(sx, sy) or k[0, 0] >= 0 or k[1, 1] >= 0:
            raise ValueError("Unsupported camera convention")
        ct = s["vol/pixels"][:].astype(np.float32)
        support = np.isin(s["vol-seg/image/pixels"][:], [1, 2, 3, 4, 7])
        affine = np.eye(4)
        affine[:3, :3] = s["vol/dir-mat"][:] @ np.diag(s["vol/spacing"][:].ravel())
        affine[:3, 3] = s["vol/origin"][:].ravel()
        camera_to_volume = s[f"projections/{args.frame}/gt-poses/cam-to-pelvis-vol"][:]
    if ct.min() >= 0:
        raise ValueError("Expected original HU volume including air")
    density = np.maximum(1 + ct / abs(float(ct.min())), 0) * support
    sc = np.diag([-1., -1., -1., 1.])
    sv = np.diag([-1., 1., 1., 1.])
    anatomy = AnatomyCT("pelvis", torch.from_numpy(density.copy())[None, None],
                        torch.tensor(sv @ affine, dtype=torch.float32)[None])
    anatomy.set_model_matrix(torch.tensor(sc @ extrinsic @ np.linalg.inv(camera_to_volume) @ sv,
                                         dtype=torch.float32)[None])
    focal = abs(k[0, 0]) * sx
    camera = Camera("deepfluoro", (0, 0, focal), (0, 0, -1), (0, 1, 0),
                    cols * sx, rows * sy,
                    (k[0, 2] + .5 - cols / 2) * sx,
                    -(k[1, 2] + .5 - rows / 2) * sy, focal)
    scene = Scene()
    scene.add_anatomies(anatomy)
    scene.add_cameras(camera)
    renderer = CTRenderer(device=torch.device(args.device))
    renderer.bind_scene(scene)
    with torch.no_grad():
        image = renderer.render(width_pixels_num=args.resolution,
                                height_pixels_num=args.resolution, binary=False,
                                efficient=True, scale_output=False, decay_type=args.mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, image[0].cpu().numpy(), cmap="gray")


if __name__ == "__main__":
    main()
