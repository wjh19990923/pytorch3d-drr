# README image provenance

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
