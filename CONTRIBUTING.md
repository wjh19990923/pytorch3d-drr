# Development and verification

Install the package with `python -m pip install -e .` after installing a
compatible PyTorch3D build. Follow AGENTS.md for the same checks used by agents.

The CPU CI job builds PyTorch3D v0.7.9 from source against PyTorch 2.5.1 on
Linux/Python 3.11. It runs the unit tests, offline rendering demo and wheel build.
It does not validate CUDA, every operating system, or real medical data.
Build failures must remain visible; tests are not skipped when PyTorch3D fails.

When changing geometry, add an analytical projection test and an image
regression assertion. When changing differentiation, test the relevant input's
gradient; a density-only test is not proof of pose gradients.
When changing the CLI, preserve JSON stdout and meaningful failure codes.

`None`/`auto` device selection may fall back to CPU. Explicit `cuda` now fails
if unavailable; this is an intentional safety change from the initial preview.
