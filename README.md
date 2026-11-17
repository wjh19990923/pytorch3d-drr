# pytorch3d-drr

**Differentiable CT projections and STL silhouettes with a shared PyTorch scene/camera API.**

[![CPU CI](https://github.com/wjh19990923/pytorch3d-drr/actions/workflows/cpu.yml/badge.svg)](https://github.com/wjh19990923/pytorch3d-drr/actions/workflows/cpu.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Renderer code for our ISBI 2027 submission
*A PyTorch3D-Based Differentiable X-ray Renderer for 2D/3D Registration*.

[Install](#install) · [Quick start](#quick-start) · [Geometry](docs/geometry.md) · [Reproducing the paper](#reproducing-the-paper)

![Pelvis DRR, SiPla natural-knee femur DRR and femoral/tibial implant silhouettes](docs/assets/real-renderers.png)

*Saved native outputs: DeepFluoro pelvis, SiPla natural-knee femur and CAMS-Knee
implants. Renderings only; no input radiographs or manuscript figures.
Contrast and crop differ across panels. [Provenance](docs/assets/README.md).*

| Renderer | Output |
|---|---|
| CT / `CTRenderer`, linear | Sampled density path integral |
| CT / `CTRenderer`, exp | `1 - exp(-0.0192 * integral)`, not transmitted intensity |
| STL / `STLRenderer` | Soft alpha silhouette, not mesh X-ray attenuation |

Research software, not a clinical device or an official PyTorch3D release.
This repository contains the renderers; see [Reproducing the paper](#reproducing-the-paper)
for the registration experiments.

## Install

Requires Python ≥3.10, PyTorch ≥2.1 and **compiled PyTorch3D**, plus NumPy,
nibabel and numpy-stl. Local CPU checks use Python 3.11 / torch 2.5.1 /
PyTorch3D 0.7.9. CPU works for the small demo; CUDA needs matching builds.

Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
first, then:

```sh
git clone https://github.com/wjh19990923/pytorch3d-drr.git
cd pytorch3d-drr
python -m pip install -e ".[examples]"
```

[Full Ubuntu CPU recipe, CUDA/Windows notes and troubleshooting](docs/usage.md#installation).
These are observed configurations, not an all-platform support guarantee.

## Quick start

No dataset or network access is needed at runtime:

```sh
python -m pytorch3d_drr doctor --json --device cpu
python -m pytorch3d_drr demo --device cpu --output outputs/demo
```

Expect `"ok": true`, `"actual_device": "cpu"`, three `(1,32,32)` arrays,
previews and `report.json`. The cube is an installation self-test.
Compare raw `.npy` arrays, not independently normalized previews.

## Python example

Small CPU example; its synthetic geometry is unrelated to the gallery.

```python
import torch
from pytorch3d_drr import (
    AnatomyCT, Camera, CubeCT, CubeSTL, CTRenderer, STLRenderer, Scene,
)

def project(anatomy, renderer, **options):
    scene = Scene()
    scene.add_anatomies(anatomy)
    scene.add_cameras(Camera(
        "demo", (0, 0, 400), (0, 0, -1), (0, 1, 0),
        160., 160., 0., 0., 400.,
    ))  # detector at z=400 mm; source at the origin
    renderer.bind_scene(scene)
    return renderer.render(width_pixels_num=32, height_pixels_num=32,
                           scale_output=False, **options)

with torch.no_grad():  # Remove for an autograd-based application.
    density, affine = CubeCT.create_voxel_grid(
        40, 30, 20, (16, 16, 16), (0, 0, 200), 1.,
    )
    volume = AnatomyCT("phantom", density, affine)
    linear = project(volume, CTRenderer(device="cpu"),
                     binary=False, efficient=False, decay_type="linear")
    exp = project(volume, CTRenderer(device="cpu"),
                  binary=False, efficient=False, decay_type="exp")
    silhouette = project(
        CubeSTL(width=40, height=30, depth=20, center_pose=(0, 0, 200)),
        STLRenderer(device="cpu"),
    )
print(linear.shape, exp.shape, silhouette.shape)  # each: (1,32,32)
```

## Real inputs and refinement

- [STL + calibration/pose JSON](examples/surface_implant.py)
- [DeepFluoro CT + specimen/frame](examples/volume_deepfluoro.py)
- [Input formats and commands](examples/README.md)
- [Geometry contract](docs/geometry.md): millimetres, column-vector model-to-world
  matrices, volume `(B,1,D,H,W)`, image `(B,H,W)`.

Connect the projection tensor to your loss and optimizer, with your own pose
parameterization and rotation centre. The gallery above shows saved renderings
from our experiments; it is not an accuracy benchmark.

## Reproducing the paper

| Paper table | Experiment | Data | In this repository |
|---|---|---|---|
| Table 1 | Implant refinement after JTML/NFD initialization, 130 images | [CAMS-Knee](https://cams-knee.orthoload.com/) (provider terms apply) | Surface renderer (`STLRenderer`); JTML/NFD initialization not included |
| Table 2 | Natural-knee perturbation recovery, 25 subjects | Laboratory data, not public | Volume renderer (`CTRenderer`); data cannot be shared |
| Table 3 | DeepFluoro registration from shared XVR initial poses, 366 images | [DeepFluoro](https://doi.org/10.7281/T1/IFSXNV) (public) | Volume renderer; similarity metrics, multiscale optimizer and evaluation scripts planned for the next release |
| Table 4 | DeepFluoro reference recovery from random starts | DeepFluoro (public) | As for Table 3 |

The exponential mapping in the paper (Eq. 2, μ = 0.0192 mm⁻¹) is
`decay_type="exp"`; the linear mapping is `decay_type="linear"`.

## Development

Implementation: [package entry point](src/pytorch3d_drr/__init__.py) ·
[geometry contract](docs/geometry.md) · [contribution guide](CONTRIBUTING.md) ·
[CPU CI](.github/workflows/cpu.yml) · [notes for coding agents](AGENTS.md).

```sh
python -m unittest discover -s tests -v
python tools/check_readme.py
python tools/check_candidate.py
```

CLI stdout is JSON; diagnostics go to stderr. Exit codes: **0** success,
**1** runtime/diagnostic failure, **2** invalid arguments. Check `ok`,
`actual_device` and files. Explicit CUDA requests fail rather than silently
falling back. `doctor` checks the environment; `demo` tests rendering.

## License and data

[MIT](LICENSE) for original code; [BSD attribution](THIRD_PARTY_NOTICES.md) for
upstream portions. Source CTs, radiographs and implant meshes are not bundled.
Dataset/CAD rights remain with their owners.
[Scope and limitations](docs/usage.md#scope-and-limitations).

## Citation

A citation for the paper will be added after the review process. This renderer
builds on [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
(Ravi et al., *Accelerating 3D Deep Learning with PyTorch3D*, arXiv:2007.08501, 2020).
