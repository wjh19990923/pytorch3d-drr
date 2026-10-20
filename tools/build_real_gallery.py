"""Plot historical renderer outputs only; no rendering or patient data export.

Pass the private experiment 014 images.npz and experiment 119 output directory.
Only the explicitly named renderer arrays are read; target radiographs are not used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bone", type=Path)
    parser.add_argument("implant", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    provenance = []
    specs = [(args.bone, "009__Ours", "CT volume / pelvis", "DeepFluoro 17-1882 · frame 009"),
             (args.bone, "050__Ours", "CT volume / another view", "DeepFluoro 17-1882 · frame 050"),
             (args.implant / "0_femur.npz", "silhouette", "STL surface / femoral implant", "CAMS-Knee · SIGM_02"),
             (args.implant / "0_tibia.npz", "silhouette", "STL surface / tibial implant", "CAMS-Knee · SIGM_02")]
    for path, key, title, subtitle in specs:
        with np.load(path, allow_pickle=False) as saved:
            array = saved[key].copy()
        assert array.ndim == 2 and np.isfinite(array).all() and array.max() > array.min()
        bounds = [0, array.shape[0], 0, array.shape[1]]
        if key == "silhouette":
            ys, xs = np.where(array > 0.01)
            bounds = [max(0, int(ys.min())-25), min(array.shape[0], int(ys.max())+26),
                      max(0, int(xs.min())-25), min(array.shape[1], int(xs.max())+26)]
        y0, y1, x0, x1 = bounds
        panels.append((array[y0:y1, x0:x1], title, subtitle))
        provenance.append(dict(source_file=path.name, key=key, shape=list(array.shape),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               display_crop_yxyx=bounds, title=title, case=subtitle))
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="#101820")
    for ax, (array, title, subtitle) in zip(axes, panels):
        ax.set_facecolor("black")
        ax.imshow(array, cmap="gray", vmin=float(array.min()), vmax=float(array.max()), interpolation="nearest")
        ax.set_title(title + "\n" + subtitle, fontsize=11, color="white", pad=14)
        ax.axis("off")
    fig.text(.5, .06, "Saved native renderer outputs · CT contrast scaled per panel · surface crops shown larger · not a scale or accuracy comparison",
             ha="center", color="#bacbd6", fontsize=11)
    fig.subplots_adjust(left=.02, right=.98, top=.87, bottom=.15, wspace=.08)
    fig.savefig(args.output / "real-renderers.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    (args.output / "real-renderers.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
