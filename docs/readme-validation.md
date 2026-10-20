# README verification — 2026-10-20

Scope: documentation, one synthetic gallery, gallery/check scripts, and CI.
No renderer implementation, research data or manuscript artwork changed.

Local verification used Windows, Python3.11.15, torch2.5.1 (CUDA11.8 build),
PyTorch3D0.7.9, NumPy2.0.1. All rendering checks explicitly used CPU. The source
checkout was exposed through PYTHONPATH; this was not a clean-environment install.

| Command/check | Result |
|---|---|
| `python -m pytorch3d_drr doctor --json --device cpu` | `ok:true`, actual device CPU |
| `python tools/build_readme_gallery.py` | Existing synthetic arrays plotted; no new rendering |
| `python tools/check_readme.py` | Relative links and actual Python example passed; all3 outputs finite/nonconstant(1,32,32), exp formula checked |
| `python -m unittest discover -s tests -v` | 6/6 passed after relocating TEMP/TMP into ignored outputs directory |
| `python tools/check_candidate.py` | 13 Python files parsed; no sensitive-pattern findings |
| Python AST parse of tools | Passed |
| `setuptools.build_meta.build_wheel('outputs/readme-wheel')` | Wheel built successfully |
| Gallery visual inspection | All3 labeled panels and profiles legible; separate scales explicit |

Initial CLI invocation without installation/PYTHONPATH correctly failed with a
missing-package error. Initial unittest run encountered a sandbox system-temp
PermissionError; setting TEMP/TMP to the checkout's ignored outputs directory
resolved it without changing tests. An optional pip-wheel frontend invocation
stalled and was interrupted; direct setuptools wheel construction then passed.
These checks do not certify clean installation, CUDA execution, real-data
alignment, clinical accuracy or future CI success.

Existing gallery input SHA256:

- linear: `e9d2044a084c98643e6dab8da57d254f6219ca0ba7f4b2414bd0503c767751ad`
- exp: `31a9183591a8708ee5a3922e79bc2fc7da9c010aa18d86b17ab109e7bc84782f`
- surface: `22b384b72a62fa049b1a5ad3d441dae2fe705b21a85da2e233d5b3b5a46f998a`

Review: documentation checks passed with no blocking findings. Full pose-refiner
API remains absent and is explicitly not advertised as packaged functionality.
