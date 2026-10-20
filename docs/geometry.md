# Geometry contract

This documents the current native API, not DiffDRR's pose convention.
Do not copy poses between the two without a calibrated conversion.

| Quantity | Contract |
|---|---|
| Physical positions, lengths, translations | millimetres |
| Anatomy model matrix | `(B,4,4)`, column-vector model-to-world |
| Matrix composition | rightmost transform is applied first |
| Volume tensor | `(B,1,D,H,W)`; array indices are z,y,x |
| Volume affine | maps x,y,z voxel indices to physical model coordinates |
| Output image | `(B,H,W)`; not `(B,1,H,W)` |
| Surface output | alpha silhouette, not an attenuation integral |
| Volume linear | sampled density path integral |
| Volume exp | `1-exp(-0.0192*linear)` |

Use `scale_output=False` when comparing physical outputs. Display scaling is
not the rendering model. `binary=False` preserves grayscale volume output.
The small offline demo uses full-grid rendering (`efficient=False`).

## Pose and rotation centre

For model point `p`, `p_world = R @ p + t`.
The renderer accepts matrices, not degree/radian Euler vectors. A rotation
matrix alone rotates around the model origin, **not automatically the centroid**.
To rotate around a specified model-space centre `c`, use translation
`c - R @ c + delta_translation`. To apply this local correction to an existing
model-to-world pose `T0`, use `T0 @ correction`. A world-space correction instead
uses left multiplication and a centre expressed in world coordinates.

Do not infer an Euler convention from the renderer name. If an upstream
optimizer uses degrees/ZXY/centroid rotation, it must perform that conversion
explicitly. The full pose-refiner API is not yet included.

## Camera

`screen_center_poses` is the detector centre, not the X-ray source.
`screen_normals` points from detector toward source.
`screen_verticals` is perpendicular to the normal.
Both direction vectors are normalized by the camera.

Let centre `d`, normal `n`, vertical `v`, principal offsets `h0,v0`, and focal
length `f` be the supplied values:

```text
principal point = d + h0 * cross(v,n) + v0 * v
source          = principal point + f * n
camera +z       = -n
camera +y       = v
camera +x       = cross(v,-n)
```

Internally PyTorch3D uses row-vector transforms in some APIs; do not transpose
the public model matrix to imitate those internals.

## Verifiable synthetic example

The demo places a 40×30×20 mm object at `(0,0,200)`, a 160×160 mm detector at
`(0,0,400)`, normal `(0,0,-1)`, vertical `(0,1,0)`, and focal length 400 mm.
Thus the source is `(0,0,0)`. The object is halfway along the central ray.
Its centre projects to NDC `(0,0)`. A 10 mm horizontal offset at z=200
has projected magnitude 20 mm, or 0.25 in detector NDC.

`test_camera_contract` checks these values independently of image appearance.
Array display flips, voxel-centre conventions and dataset orientation must
still be verified for real-data adapters. Synthetic checks do not establish
clinical alignment or registration accuracy.
