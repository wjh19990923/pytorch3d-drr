# External-data examples

## CAMS-Knee implant surface

Data source: https://cams-knee.orthoload.com/
ETH overview: https://movement.ethz.ch/data-repository/cams-knee-project.html

The overview identifies a freely downloadable example trial. Full datasets and
implant geometry can have separate terms. We do not bundle or rehost them.
The three historical local samples have not yet been mapped to the official
download's file identifiers in this candidate.

Create a JSON file containing:

- `camera`: keyword arguments for `Camera`, excluding `names`.
  `screen_center_poses`, `screen_normals`, `screen_verticals` are three-vectors;
  `screen_sizes_h`, `screen_sizes_v`, `principal_points_h`,
  `principal_points_v`, `focal_lengths` are millimetres.
- `model_to_world`: a 4 by 4 column-vector transform.

Use actual dataset calibration and pose, not fabricated values.

```sh
python examples/surface_implant.py --stl /path/femoral.stl --geometry /path/calibration.json --output outputs/surface.png --device cuda
```

## DeepFluoro volume

Dataset DOI: https://doi.org/10.7281/T1/IFSXNV
Source project: https://github.com/rg2/DeepFluoroLabeling-IPCAI2020

The example reads the original HDF5 layout, uses the saved camera-to-pelvis
pose, and renders the pelvis support labels 1,2,3,4,7. It uses the native detector
field of view, without the experimental crop. Display orientation is raw detector
orientation; `rot-180-for-up` is not applied. This is a rendering example, not
the XVR registration pipeline or an accuracy evaluation.

```sh
python examples/volume_deepfluoro.py --data /path/ipcai_2020_full_res_data.h5 --specimen 17-1882 --frame 000 --mode linear --output outputs/linear.png --device cuda
python examples/volume_deepfluoro.py --data /path/ipcai_2020_full_res_data.h5 --specimen 17-1882 --frame 000 --mode exp --output outputs/exp.png --device cuda
```

Real-data examples require a separate forward-render alignment check before
release. They have not been run as new GPU experiments during packaging.
