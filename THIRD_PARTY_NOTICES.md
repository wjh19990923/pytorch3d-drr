# Third-party attribution

Original project contributions are licensed under MIT by Jinhao Wang.
This does not relicense third-party portions.

## PyTorch3D

Repository: https://github.com/facebookresearch/pytorch3d
License source: https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/LICENSE

`src/pytorch3d_drr/raysampling_generalized.py` adapts PyTorch3D ray sampling
and retains its original Facebook copyright header. The generalized
raymarcher also follows the PyTorch3D absorption-only raymarcher API and
documentation. These upstream portions remain subject to the BSD license
provided in `licenses/PYTORCH3D-BSD.txt`. Historical Facebook notices are
retained alongside the current upstream Meta license text.

This project adds generalized/partial sampling, density integration,
linear/exp output and scene/camera handling. Its depth-bound interpolation
has been changed to preserve autograd.

PyTorch, PyTorch3D, NumPy, NiBabel, numpy-stl and optional h5py/matplotlib
are external dependencies, not vendored libraries. Their respective licenses
continue to apply. No affiliation with or endorsement by Meta is implied.

## Datasets

CAMS-Knee and DeepFluoro are referenced, not distributed. Their image and
geometry licenses are separate from this software license. Users must obtain
data through the relevant providers and comply with their terms.
