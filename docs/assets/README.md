# README image provenance

## Real-data gallery

`real-renderers.png` is a new documentation layout of **existing saved native
renderings**, not a manuscript figure or a new rendering experiment.

| Panels | Historical source | Array / resolution |
|---|---|---|
| Pelvis view | Experiment 014, sourced from 002 renderer alignment; DeepFluoro specimen 17-1882, frame 009 | `images.npz`, `009__Ours`; 256 x 256 |
| Natural knee (femur only) | Experiment 038, run 038_20261006T075720Z; SiPla SUBN_02 / dkb_01 / image001, original CSV pose | `images_private.npz`, `reference_ours`; 128 x 128, native efficient linear rendering |
| Femoral / tibial implant | Experiment 119, original alignment check 20261014T111105Z; SIGM_02, C_SIGM_02_ch_sit_01_001 | `0_femur.npz` / `0_tibia.npz`, `silhouette`; 512 x 512 |

Only the author's renderer outputs are selected. No target X-rays, other
renderers' outputs, source CTs or STL files are exported. These assets precede
the manuscript layouts. The implant example is the original SIGM_02 check,
not the later ATTM07/GOLF03/PSLE04 manuscript examples. The bone images are
historical 014 and 038 outputs, not the manuscript's 090 or renderer-branches assets.

CT panels use their saved values with per-panel min/max display scaling.
Surface panels show soft alpha silhouettes, **not attenuation or shaded 3D
surfaces**. Their bounding boxes use alpha > 0.01 with a 25-pixel margin.
Nearest-neighbour display preserves the saved pixels; no smoothing, AI image
generation, pose fitting or intensity-model conversion is applied. Panel sizes
are not a physical scale comparison. Exact input SHA256 values and crops are in
[real-renderers.json](real-renderers.json).

To rebuild with authorized copies of the historical output archives:

```sh
python tools/build_real_gallery.py /path/014/images.npz /path/119/20261014T111105Z --sipla /path/038/images_private.npz
```

The private arrays are deliberately not bundled. Dataset attribution:
[DeepFluoro](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) and
[CAMS-Knee](https://cams-knee.orthoload.com/). Dataset and implant-design rights
remain with their owners; the code's MIT license does not relicense underlying
datasets or CAD designs. Obtain source data under the providers' terms.

## Offline synthetic self-test

`synthetic-renderers.png` plots the existing `outputs/demo/{linear,exp,surface}.npy`
fixtures from this repository's native offline demo. They are synthetic box
projections, not medical data or manuscript assets. No research-results or
manuscript directories are read by the gallery tool.

The saved demo uses CPU,32x32 pixels, a40x30x20mm box centred at(0,0,200),
a160x160mm detector at(0,0,400) and a source at the origin. Display uses raw values,
separate color scales, nearest-neighbour interpolation and central-row profiles.
No image enhancement or new rendering is needed to build the gallery.

To recreate the fixtures and presentation from the clone:

```sh
python -m pytorch3d_drr demo --device cpu --output outputs/demo
python tools/build_readme_gallery.py
```

The gallery tool needs NumPy/matplotlib (`.[examples]`) and prints input SHA256
values. Floating-point bytes can differ across dependency versions. This is not
an accuracy benchmark. Original documentation artwork uses the project's MIT license.
The SiPla panel shows natural femoral bone, not a whole-knee or implant render.
It uses the original CSV-pose projection, not a selected successful registration.
The source experiment's failed registrations are not hidden or presented as
successes: this gallery only illustrates rendering capability. No original
SiPla pixels or volume data are distributed. No new rendering was run.
