"""Small deterministic phantom; not a clinical accuracy benchmark."""
import json
from pathlib import Path
import time


def run_demo(output, device="cpu", resolution=32):
    if not 8 <= resolution <= 128:
        raise ValueError("Demo resolution must be between 8 and 128")
    import numpy as np
    import torch
    from .scene_tools import AnatomyCT, CubeCT, CubeSTL, Camera, Scene
    from .pytorch3d_renderer import CTRenderer, STLRenderer
    from .utilities import determine_device
    selected = determine_device(device)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "ok": True, "synthetic": True,
              "actual_device": str(selected), "resolution": resolution, "images": {}}
    for mode in ("linear", "exp", "surface"):
        if mode == "surface":
            anatomy = CubeSTL(width=40, height=30, depth=20, center_pose=(0, 0, 200))
            renderer = STLRenderer(device=selected)
        else:
            density, affine = CubeCT.create_voxel_grid(40, 30, 20, (16, 16, 16), (0, 0, 200), 1.)
            anatomy = AnatomyCT("phantom", density, affine)
            renderer = CTRenderer(device=selected)
        scene = Scene()
        scene.add_anatomies(anatomy)
        scene.add_cameras(Camera("synthetic", (0, 0, 400), (0, 0, -1), (0, 1, 0),
                                 160., 160., 0., 0., 400.))
        renderer.bind_scene(scene)
        start = time.perf_counter()
        with torch.no_grad():
            kwargs = dict(width_pixels_num=resolution, height_pixels_num=resolution, scale_output=False)
            if mode != "surface":
                kwargs.update(binary=False, efficient=False, decay_type=mode)
            array = renderer.render(**kwargs).detach().cpu().numpy()
        if array.shape != (1, resolution, resolution) or not np.isfinite(array).all() or np.ptp(array) <= 0:
            raise RuntimeError(f"{mode}: expected finite nonconstant (1,H,W) output")
        np.save(folder / f"{mode}.npy", array)
        pixels = np.round((array[0] - array.min()) / np.ptp(array) * 255).astype(np.uint8)
        with (folder / f"{mode}.pgm").open("wb") as stream:
            stream.write(f"P5\n{resolution} {resolution}\n255\n".encode("ascii"))
            stream.write(pixels.tobytes())
        report["images"][mode] = {"shape": list(array.shape), "min": float(array.min()),
                                 "max": float(array.max()), "seconds": time.perf_counter() - start,
                                 "array": str((folder / f"{mode}.npy").resolve()),
                                 "preview": str((folder / f"{mode}.pgm").resolve())}
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
