import unittest
import torch
from types import SimpleNamespace
from pytorch3d_drr.raymarching_generalized import AbsorptionOnlyRaymarcherGeneralized


class CoreTests(unittest.TestCase):
    def test_modes_and_density_gradient(self):
        density = torch.ones(1, 2, 2, 3, 1, requires_grad=True)
        bundle = SimpleNamespace(
            lengths=torch.arange(3.).expand(1, 2, 2, 3),
            directions=torch.tensor([0., 0., 1.]).expand(1, 2, 2, 3))
        linear = AbsorptionOnlyRaymarcherGeneralized("linear")(density, ray_bundle=bundle)
        exp = AbsorptionOnlyRaymarcherGeneralized("exp")(density, ray_bundle=bundle)
        torch.testing.assert_close(linear, torch.full_like(linear, 3))
        torch.testing.assert_close(exp, 1 - torch.exp(-0.0192 * linear))
        linear.sum().backward()
        torch.testing.assert_close(density.grad, torch.ones_like(density))


if __name__ == "__main__":
    unittest.main()
