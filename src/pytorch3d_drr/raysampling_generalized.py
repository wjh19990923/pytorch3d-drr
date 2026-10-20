# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import importlib.util

pytorch3d_spec = importlib.util.find_spec("pytorch3d")
if pytorch3d_spec is not None:
    from pytorch3d.renderer.cameras import CamerasBase
    from pytorch3d.renderer.implicit import RayBundle

    print("Pytorch3d detected, importing packages.")
else:
    print("PyTorch3D not found. Skipping imports related to PyTorch3D.")


"""
In this generalized version we allow sampler to have different depths for different anatomy.
During using it, min_depth and max_depth is chosen based on the anatomies size and their transformation matrix.
This feature helps us to reduce memory cost significantly during rendering multi anatomies.
In the Future we allow sampler to have different x and y values.
"""


class GridRaysamplerGeneralized(torch.nn.Module):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined
    minimum and maximum depth.
    The raysampler first generates a 3D coordinate grid of the following form:
    ```
       / min_x, min_y, max_depth -------------- / max_x, min_y, max_depth
      /                                        /|
     /                                        / |     ^
    / min_depth                    min_depth /  |     |
    min_x ----------------------------- max_x   |     | image
    min_y                               min_y   |     | height
    |                                       |   |     |
    |                                       |   |     v
    |                                       |   |
    |                                       |   / max_x, max_y,     ^
    |                                       |  /  max_depth        /
    min_x                               max_y /                   / n_pts_per_ray
    max_y ----------------------------- max_x/ min_depth         v
              < --- image_width --- >
    ```
    In order to generate ray points, `GridRaysampler` takes each 3D point of
    the grid (with coordinates `[x, y, depth]`) and unprojects it
    with `cameras.unproject_points([x, y, depth])`, where `cameras` are an
    additional input to the `forward` function.
    Note that this is a generic implementation that can support any image grid
    coordinate convention. For a raysampler which follows the PyTorch3D
    coordinate conventions please refer to `NDCGridRaysampler`.
    As such, `NDCGridRaysampler` is a special case of `GridRaysampler`.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: List[float],
        max_depth: List[float],
    ) -> None:
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's source pixel's center.
            max_x: The rightmost x-coordinate of each ray's source pixel's center.
            min_y: The topmost y-coordinate of each ray's source pixel's center.
            max_y: The bottommost y-coordinate of each ray's source pixel's center.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: List of The minimum depth of a ray-point. Its length determines batch size
            max_depth: List The maximum depth of a ray-point. Its length determines batch size
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        assert len(min_depth) == len(max_depth), "min_depth and max_depth must have the same size"
        # get the initial grid of image xy coords
        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
                        torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
                    )
                )
            ),
            dim=-1,
        )
        self.register_buffer("_xy_grid", _xy_grid, persistent=False)

    def forward(self, cameras: "CamerasBase", **kwargs) -> "RayBundle":
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, image_height, image_width, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, image_height, image_width, 2)`
                containing the 2D image coordinates of each ray.
        """
        from pytorch3d.renderer.cameras import CamerasBase
        from pytorch3d.renderer.implicit import RayBundle

        batch_size = cameras.R.shape[0]
        assert len(self._min_depth) == batch_size

        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device)[None].expand(batch_size, *self._xy_grid.shape)

        return _xy_to_ray_bundle_generalized(cameras, xy_grid, self._min_depth, self._max_depth, self._n_pts_per_ray)


class NDCGridRaysamplerGeneralized(GridRaysamplerGeneralized):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.
    `NDCGridRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the border of the leftmost / rightmost / topmost / bottommost pixel
    has coordinates 1.0 / -1.0 / 1.0 / -1.0 respectively.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: List[float],
        max_depth: List[float],
    ) -> None:
        """
        Args:
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        half_pix_width = 1.0 / image_width
        half_pix_height = 1.0 / image_height
        super().__init__(
            min_x=1.0 - half_pix_width,
            max_x=-1.0 + half_pix_width,
            min_y=1.0 - half_pix_height,
            max_y=-1.0 + half_pix_height,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )


class PartialNDCGridRaysampler(torch.nn.Module):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.
    `NDCGridRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the border of the leftmost / rightmost / topmost / bottommost pixel
    has coordinates 1.0 / -1.0 / 1.0 / -1.0 respectively.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        w_start_idx: int,
        w_end_idx: int,
        h_start_idx: int,
        h_end_idx: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            w_start_idx: left index (starting index) in horizontal size of image grid.
               shape (batch_size, 1)
            w_end_idx: right index (end index) in horizontal size of image grid.
            h_start_idx: top index (starting index) in vertical size of image grid.
            h_end_idx: bottom index (end index) in vertical size of image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        :: note.. for a batch, we use the largest partial image for the whole patch (so images of
        size (100x100, 3x3) will give (100x100, 100x100), starting from the lower idx - unless we
        cross the maximal value. this procedure is not optimal if you have small partial images and
        large in the same patch but simple to vectorize
        :: warning.. not vectorized properly
        :: warning.. needs proper batched testcases of anatomies with different sizes and out of the
        image
        """
        super().__init__()
        assert (w_end_idx >= w_start_idx).all()
        assert (h_end_idx >= h_start_idx).all()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        assert w_start_idx.shape == w_end_idx.shape == h_start_idx.shape == h_end_idx.shape
        batchsize = w_start_idx.shape[0]

        half_pix_width = 1.0 / image_width
        half_pix_height = 1.0 / image_height
        min_x = 1.0 - half_pix_width
        max_x = -1.0 + half_pix_width
        min_y = 1.0 - half_pix_height
        max_y = -1.0 + half_pix_height
        """
        # get the initial grid of image xy coords
        # and cut out the part that is given by the indices (derived from the anatomy extents)
        # this version is not vectorized properly but readable
        always cut the largest imageize in the batch
        """
        maxHeight = torch.max(h_end_idx - h_start_idx)
        maxWidth = torch.max(w_end_idx - w_start_idx)

        def check_idx_limits(low: torch.Tensor, high: torch.Tensor, minVal: int, maxVal: int):
            """force low and height to be within limits but retain their length
            all parameters are shape (batchsize,)
            low = (1,4,9)
            high= (10, 8, 14)
            minVal = 3
            maxVal = 13
            will give: (3, 4, 8) and (12, 8, 13)
            """
            diff = high - low
            assert (torch.abs(diff) <= (maxVal - minVal)).any(), "Cannot fit all elements between min and maxVal"
            newLow = low.clone()
            newHigh = high.clone()
            toolowIdx = low < minVal
            newLow[toolowIdx] = minVal
            newHigh[toolowIdx] = (minVal + diff)[toolowIdx]
            tooHighIdx = high > maxVal
            newHigh[tooHighIdx] = maxVal
            newLow[tooHighIdx] = (maxVal - diff)[tooHighIdx]
            return newLow, newHigh
        hStart, hEnd = check_idx_limits(h_start_idx, h_start_idx + maxHeight, 0, image_height)
        hIdx = torch.stack(tuple(torch.arange(hStart[i], hEnd[i]) for i in range(batchsize))).to(torch.int64)
        hNDC = torch.gather(
            torch.linspace(min_y, max_y, image_height, dtype=torch.float32).expand(batchsize, -1),
            dim=1,
            index=hIdx,
        )

        wStart, wEnd = check_idx_limits(w_start_idx, w_start_idx + maxWidth, 0, image_width)
        wIdx = torch.stack(tuple(torch.arange(wStart[i], wEnd[i]) for i in range(batchsize))).to(torch.int64)
        wNDC = torch.gather(
            torch.linspace(min_x, max_x, image_width, dtype=torch.float32).expand(batchsize, -1),
            dim=1,
            index=wIdx,
        )

        # (B, H, W, 2)
        _xy_grid = torch.stack(
            tuple(
                torch.stack(tuple(reversed(torch.meshgrid(hNDC[i], wNDC[i]))), dim=-1)
                for i in range(batchsize)
            )
        )
        """old version
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, image_height, dtype=torch.float32)[h_start_idx: h_end_idx + 1],
                        torch.linspace(min_x, max_x, image_width, dtype=torch.float32)[w_start_idx: w_end_idx + 1],
                    )
                )
            ),
            dim=-1,
        """
        assert _xy_grid.shape == (batchsize, hNDC.shape[1], wNDC.shape[1], 2)
        self.register_buffer("_xy_grid", _xy_grid, persistent=False)
        self.w_start_idx = wStart
        self.w_end_idx = wEnd
        self.h_start_idx = hStart
        self.h_end_idx = hEnd

    def forward(self, cameras: "CamerasBase", **kwargs) -> "RayBundle":
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, image_height, image_width, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, image_height, image_width, 2)`
                containing the 2D image coordinates of each ray.
        """
        from pytorch3d.renderer.cameras import CamerasBase
        from pytorch3d.renderer.implicit import RayBundle

        assert cameras.R.shape[0] == self._xy_grid.shape[0], "Mismatching batchsize"
        device = cameras.device
        xy_grid = self._xy_grid.to(device)
        return _xy_to_ray_bundle_generalized(
            cameras, xy_grid, self._min_depth, self._max_depth, self._n_pts_per_ray
        )
        """ old
        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device)[None].expand(batch_size, *self._xy_grid.shape)

        return _xy_to_ray_bundle_generalized(cameras, xy_grid, [self._min_depth], [self._max_depth], self._n_pts_per_ray)
        """


def _xy_to_ray_bundle_generalized(
    cameras: "CamerasBase",
    xy_grid: torch.Tensor,
    min_depth: List[float],
    max_depth: List[float],
    n_pts_per_ray: int,
) -> "RayBundle":
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.
    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    FV: seems to follow closely the pytorch3d implementation
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/
        renderer/implicit/raysampling.html?highlight=NDCMultinomialRaysampler%20
    """
    from pytorch3d.renderer.cameras import CamerasBase
    from pytorch3d.renderer.implicit import RayBundle

    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore
    # ray z-coords
    # Preserve pose gradients through both ray-depth bounds.
    lower = torch.as_tensor(min_depth, dtype=xy_grid.dtype, device=xy_grid.device).reshape(-1, 1)
    upper = torch.as_tensor(max_depth, dtype=xy_grid.dtype, device=xy_grid.device).reshape(-1, 1)
    alpha = torch.linspace(0, 1, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device)[None]
    depths = torch.lerp(lower, upper, alpha)
    rays_zs = depths[:, None, :].expand(batch_size, n_rays_per_image, n_pts_per_ray)

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),  # pyre-ignore
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject)  # pyre-ignore

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )
