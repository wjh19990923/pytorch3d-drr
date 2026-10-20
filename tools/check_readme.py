"""Validate relative README links and execute its no-data Python example on CPU."""
from pathlib import Path
import re
import numpy as np

root = Path(__file__).resolve().parents[1]
readme = (root / 'README.md').read_text(encoding='utf-8')
for target in re.findall(r'\]\(([^)]+)\)', readme):
    if target.startswith(('http://', 'https://', '#')):
        continue
    assert (root / target.split('#')[0]).is_file(), target
blocks = re.findall(r'```python\n(.*?)```', readme, re.S)
assert len(blocks) == 1
namespace = {}
exec(compile(blocks[0], 'README.md Python example', 'exec'), namespace)
for name in ['linear', 'exp', 'silhouette']:
    value = namespace[name].detach().cpu().numpy()
    assert value.shape == (1, 32, 32) and np.isfinite(value).all() and np.ptp(value) > 0
np.testing.assert_allclose(namespace['exp'].numpy(),
                           1 - np.exp(-.0192 * namespace['linear'].numpy()), atol=1e-6)
print('README local links and CPU Python example passed.')
