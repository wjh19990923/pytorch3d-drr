"""Render an externally supplied, licensed femoral implant mesh."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stl", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    import torch
    import matplotlib.pyplot as plt
    from pytorch3d_drr import AnatomySTL, Camera, Scene, STLRenderer
    geometry = json.loads(Path(args.geometry).read_text(encoding="utf-8"))
    camera_args = geometry["camera"]
    for key in ("screen_center_poses", "screen_normals", "screen_verticals"):
        camera_args[key] = tuple(camera_args[key])
    anatomy = AnatomySTL.load_data(args.stl)
    matrix = torch.tensor(geometry["model_to_world"], dtype=torch.float32)
    if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
        raise ValueError("model_to_world must be a finite 4x4 matrix")
    anatomy.set_model_matrix(matrix[None])
    scene = Scene()
    scene.add_anatomies(anatomy)
    scene.add_cameras(Camera(names="implant", **camera_args))
    renderer = STLRenderer(device=torch.device(args.device))
    renderer.bind_scene(scene)
    with torch.no_grad():
        image = renderer.render(width_pixels_num=args.resolution,
                                height_pixels_num=args.resolution, scale_output=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, image[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)


if __name__ == "__main__":
    main()
