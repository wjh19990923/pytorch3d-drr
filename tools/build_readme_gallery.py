"""Plot existing synthetic arrays only; no renderer or medical-data access."""
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
destination = root / 'docs/assets'
destination.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1.4]})
for col, (mode, title) in enumerate(zip(['linear', 'exp', 'surface'],
                                      ['Volume / linear', 'Volume / exp', 'Surface / alpha'])):
    path = root / 'outputs/demo' / (mode + '.npy')
    array = np.load(path, allow_pickle=False)
    assert array.shape == (1, 32, 32) and np.isfinite(array).all()
    array = array[0]
    im = axes[0, col].imshow(array, cmap='gray', interpolation='nearest', vmin=0)
    axes[0, col].set_title(title)
    axes[0, col].set_axis_off()
    fig.colorbar(im, ax=axes[0, col], fraction=.045, pad=.04)
    axes[1, col].plot(array[16], color=['#2563eb', '#059669', '#9333ea'][col])
    axes[1, col].set_xlabel('Detector column (pixel)')
    axes[1, col].set_ylabel('Raw value')
    axes[1, col].grid(alpha=.2)
    print(mode, hashlib.sha256(path.read_bytes()).hexdigest())
fig.suptitle('Synthetic box | saved 32 x 32 CPU demo outputs | no patient data', fontsize=13)
fig.tight_layout()
fig.savefig(destination / 'synthetic-renderers.png', dpi=150)
plt.close(fig)
