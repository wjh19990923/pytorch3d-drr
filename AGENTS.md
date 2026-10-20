# Agent instructions

## Start here

- Read README.md and docs/geometry.md before changing geometry.
- Editable source is `src/pytorch3d_drr/*.py`, not generated build directories.
- The core is a research preview. Do not claim clinical accuracy or benchmark
  improvements from the synthetic demo.
- Surface means silhouette, not mesh X-ray attenuation.

## Safe verification

```sh
python -m pytorch3d_drr doctor --json --device cpu
python -m unittest discover -s tests -v
python -m pytorch3d_drr demo --device cpu --output outputs/demo
python tools/check_candidate.py
```

These commands require no patient data or downloads at runtime.
Install dependencies first. Do not replace a failed dependency install with a
mock renderer and call it a passing integration test.

Use CPU by default. Do not initiate GPU experiments, download medical data,
publish data/meshes, or change historical experiments without authorization.
Do not silently fall back from requested CUDA to CPU.

## Outputs and errors

`doctor` and `demo` emit one JSON object on stdout. Import chatter and warnings
go to stderr. Success returns 0; runtime/diagnostic failure 1; invalid CLI
syntax 2 (argparse). Check `ok`, `actual_device`, and file existence.
`doctor` only checks environment imports/extension/allocation. `demo` tests
real native rendering. Neither is an accuracy benchmark.

Never commit outputs/, data, wheel/build artifacts, secrets or local paths.
Preserve MIT and third-party BSD notices. Keep the native API backward compatible
unless a breaking change is explicitly approved and documented.
