from typing import Union

import torch

"""
in this generalized version, we use the distance between rays_points
to have more realistic rendere.
we also change torch.prod to torch.sum to calculate total_transmission.
the previous one resulted the number so close to 1.0, so the img was so bright like a shadow.
in that case we needed a specific function to covert the pixel from very sharp distribution to uniform one.
"""


class AbsorptionOnlyRaymarcherGeneralized(torch.nn.Module):
    """
    Raymarch using the Absorption-Only (AO) algorithm.
    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray, 1)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light transmitted along each ray
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    """

    def __init__(self, decay: str = 'exp') -> None:
        """
        Args:
            decay (str): 'exp' or 'linear'.
        """
        super().__init__()
        assert decay in ['exp', 'linear'], "decay must be 'exp' or 'linear'"
        self.decay = decay
        # 70 keV water attenuation coefficient (mm^-1)
        self.mu_water = 0.0192

    def forward(self, rays_densities: torch.Tensor, decay: str = 'exp', **kwargs) -> Union[None, torch.Tensor]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray)` whose values  wwrange in [0, 1].
        ..note:: ray_bundle.lengths have shape (batch_size, img_height, width, n_pts_per_ray)
        Returns:
            opacities: A tensor of per-ray opacity values of shape `(..., 1)`.
                Its values range between [0, 1] and denote the total amount
                of light that has been absorbed for each ray. E.g. a value
                of 0 corresponds to the ray completely passing through a volume.

        ..note:: FV: i think we are trying to correct for the actual path length using the rays
        directions, but seems strangely implemented. ray_lengths horizontal is calculated from one
        certain ray, but i do not understand why this particular one.
        as far as i can see from teh implementation, length already contains the z-component between
        depth_min and depth_max and ignores direction of the ray
        """
        ray_bundle = kwargs.get("ray_bundle")
        ray_lengths_horizontal = torch.ones(ray_bundle.lengths.shape, device=rays_densities.device) * (
            ray_bundle.lengths[0, 0, 0, 1] - ray_bundle.lengths[0, 0, 0, 0]
        )
        ray_lengths_per_horizontal_unit = torch.linalg.vector_norm(ray_bundle.directions, dim=-1)[..., None]
        ray_lengths = ray_lengths_horizontal[..., :, None] * ray_lengths_per_horizontal_unit[..., None, :]

        rays_transmissions = (rays_densities * ray_lengths)[..., 0]
        # for real x-ray decay, water attenuation coefficient under 70keV = 0.0192/mm
        # linear or exponential attenuation
        if self.decay == 'linear':
            total_transmission = torch.sum(rays_transmissions, dim=-1, keepdim=True)
        else:  # exponential decay: I = 1 - exp(- μ * ∑ ρ·ds)
            total_transmission = 1 - torch.exp(-self.mu_water * torch.sum(rays_transmissions, dim=-1, keepdim=True))

        return total_transmission
