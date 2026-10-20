"""
After creating your scene. You just need create your renderer on your desired device (cpu, gpu).
Then call the render function of your renderer.
"""
from typing import List, Tuple, Union
import nibabel as nib
import numpy as np
import torch

# from pytorch3d.renderer.lighting import PointLights
# from pytorch3d.renderer.materials import Materials

from .raymarching_generalized import AbsorptionOnlyRaymarcherGeneralized
from .raysampling_generalized import (
    NDCGridRaysamplerGeneralized,
    PartialNDCGridRaysampler,
)

from .scene_tools import *
from .utilities import determine_device


def checkImportPytorch3D(module_name: str):
    imports = None
    import importlib.util
    pytorch3d_spec = importlib.util.find_spec("pytorch3d")
    if pytorch3d_spec is not None:
        if module_name == "VolumeRenderer":
            from pytorch3d.renderer import VolumeRenderer
            imports = VolumeRenderer
        if module_name == "MeshRenderer":
            from pytorch3d.renderer import MeshRenderer
            imports = MeshRenderer
        if module_name == "MeshRasterizer":
            from pytorch3d.renderer import MeshRasterizer
            imports = MeshRasterizer
        if module_name == "RasterizationSettings":
            from pytorch3d.renderer import RasterizationSettings
            imports = RasterizationSettings
        if module_name == "BlendParams":
            from pytorch3d.renderer.blending import BlendParams
            imports = BlendParams
        if module_name == "SoftSilhouetteShader":
            from pytorch3d.renderer import SoftSilhouetteShader
            imports = SoftSilhouetteShader

    return imports


VolumeRenderer = checkImportPytorch3D("VolumeRenderer")
MeshRasterizer = checkImportPytorch3D("MeshRasterizer")
RasterizationSettings = checkImportPytorch3D("RasterizationSettings")
BlendParams = checkImportPytorch3D("BlendParams")
SoftSilhouetteShader = checkImportPytorch3D("SoftSilhouetteShader")
MeshRenderer = checkImportPytorch3D("MeshRenderer")

eps = np.finfo(np.float32).eps
DTYPE_TORCH = torch.float32


class Renderer:
    """
    This is the base class for rendering.
    STL_renderer and CT_renderer are inherited from this class.
    """

    def __init__(self, device=None):
        self.scene = None
        self.img_pixels_num = (512, 512)
        self.device = determine_device(device=device)

    def bind_scene(self, scene: Scene = None):
        """
        this operation needs to be done before calling render() function.
        """
        if scene is not None:
            self.scene = scene.to(self.device)

    def set_resolution(self, width_pixels_num: int, height_pixels_num: int):
        """
        Use this function to determine number of pixels for rendered image.
        Not that they are not related to the width and height of intensifier or calibration pixel size
        """
        self.img_pixels_num = (width_pixels_num, height_pixels_num)

    @staticmethod
    def change_camera_extrinsic_mat_based_anatomy(anatomy: Union[AnatomySTL, AnatomyCT], camera: Camera):
        """
        Each anatomy has a tmat which is use for transforming from anatomy coordinate to world coordinate.
        For rendering an anatomy, we change camera.R and camera.T based on anatomy's tmat and camera's tmat
        """
        anatomy_to_world = anatomy.tmats
        world_to_camera = torch.linalg.inv(camera.tmats)
        # Remember tmats multiplied from the left side: transfomed_point = tmat * point, so
        anatomy_to_camera = torch.matmul(world_to_camera, anatomy_to_world)
        # Remember R multiplied from the right side
        camera.R = anatomy_to_camera[:, 0:3, 0:3].permute(0, 2, 1)
        camera.T = anatomy_to_camera[:, 0:3, 3]

    @staticmethod
    def return_back_camera_extrinsic_mat_original_value(camera: Camera):
        """
        after each rendering we return back camera.R and camera.T to their initial value when the camera was created.
        """
        camera.R = camera.world_to_camera_R
        camera.T = camera.world_to_camera_T

    def to(self, new_device: Union[str, torch.device]):
        """
        change the included scene to new device.
        By doing that all the objects in the scene will be transferred to the new device.
        And it return itself
        """
        self.device = determine_device(new_device)
        if self.scene is not None:
            self.scene = self.scene.to(new_device)
        return self

    @staticmethod
    def adjust_contrast_brightness(img, window_middle, window_length):
        """
          this function change pixel value of an image based on following diagram:
          middle is related to brightness and length is related to contrast.
          they must be between 0 and 1
          ^
          |           length
        1 |           ......______________
          |                /
          |               /
          |              /.
          |             / .
        0 |____________/__.____________ >
           0            middle         1
        """
        last_zero = window_middle - window_length / 2
        first_one = window_middle + window_length / 2
        assert last_zero > 0.0 - eps, "window_middle - window_length/ 2 should't be less than 0.0"
        assert first_one < 1.0 + eps, "window_middle + window_length/ 2 should't be more than 1.0"
        assert torch.max(img) < 1.0 + eps, "maximum pixel must be less than 1.0"
        assert torch.min(img) > 0.0 - eps, "minimum pixel must be more than 0.0"
        alpha = torch.max(torch.zeros(img.shape), (img - last_zero) / window_length)
        beta = torch.max(torch.ones(img.shape), (img - last_zero) / window_length)
        return alpha / beta


class STLRenderer(Renderer):
    """
    This class is for rendering stl_anatomies.
    For using this class, you don't need to worried about all the class's methods. We handle them.
    You just need to create your scene, create an object of renderer, then bind the scene to the renderer.
    And call the render function
    """

    def __init__(self, device=None):
        super(STLRenderer, self).__init__(device=device)
        self.sigma = 1e-7
        self.gamma = 1e-7

    def set_rasterizer(self, camera: Camera):
        blur_radius = np.log(1.0 / 1e-7 - 1.0) * self.sigma

        # to speed up the STL renderer, we need lower faces_per_pixel, None for bin_size
        # if there is warning or need to increase accuracy, faces_per_pixel=100, bin_size=0
        raster_settings = RasterizationSettings(image_size=self.img_pixels_num, blur_radius=blur_radius,
                                                faces_per_pixel=100, bin_size=None)
        rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
        return rasterizer

    def set_shader(self, camera: Camera):
        # blend_params = BlendParams(self.sigma, self.gamma, (0, 0, 0))
        blend_params_Silhouette = BlendParams(sigma=self.sigma, gamma=self.gamma)

        # lights = PointLights(
        # device=self.device,
        # location=((0.0, 1.0, 0.0),),
        # ambient_color=((1.0, 0.0, 0.0),),
        # diffuse_color=((0.0, 0.0, 0.0),),
        # specular_color=((0.0, 0.0, 0.0),),
        # )
        # materials = Materials(
        # ambient_color=((1, 1, 1),),
        # diffuse_color=((1, 1, 1),),
        # specular_color=((1, 1, 1),),
        # shininess=0,
        # device=self.device,
        # )
        # shader = SoftPhongShader(
        # lights=lights,
        # cameras=camera,
        # materials=materials,
        # blend_params=blend_params,
        # )
        # I am not sure if phone renderer actually has gradient, but soft silhoutee does.
        shader = SoftSilhouetteShader(blend_params=blend_params_Silhouette)
        return shader

    def render(
            self, cam_index: int = 0, width_pixels_num: int = 512, height_pixels_num: int = 512,
            scale_output=True
    ):
        """
        It generates a grayscale rendered image of the scene from view point of the camera given by user.
        param: width_pixels_num:
        height_pixels_num:
        param: shadow: If it is False, it will generate the image in RGB values.
        scale_output: scale to 0-1

        return: it returns a torch.tensor on self.device in the shape of:
                (batch_size, height_pixels_num, width_pixels_num, 3) when shadow is False
                (batch_size, height_pixels_num, width_pixels_num) when shadow is True
        """
        assert self.scene.anatomies_type == "STL", "Scene must only have stl anatomies"
        camera = self.scene.cameras[cam_index]
        self.set_resolution(width_pixels_num=width_pixels_num, height_pixels_num=height_pixels_num)

        rendered_img = torch.zeros(self.scene.get_batch_size(), *self.img_pixels_num, device=self.device)
        for anatomy in self.scene.anatomies:
            self.change_camera_extrinsic_mat_based_anatomy(anatomy=anatomy, camera=camera)
            rasterizer = self.set_rasterizer(camera=camera)
            shader = self.set_shader(camera=camera)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            rendered_img = torch.max(rendered_img, renderer(anatomy)[..., 3])  # maximum selecting pixel
            # rendered_img += renderer(anatomy)[..., 3] # overlapping
            self.return_back_camera_extrinsic_mat_original_value(camera=camera)

        # rendered_img = rendered_img.sum(dim=3)
        img_min = torch.min(rendered_img)
        img_max = torch.max(rendered_img)
        if scale_output:
            rendered_img = (1 / (img_max - img_min)) * (rendered_img - img_min)
        return rendered_img


class CTRenderer(Renderer):
    """
    This class is for rendering ct_anatomies.
    For using this class, you don't need to worried about all the class's methods. We handle them.
    You just need to create your scene, create an object of renderer, then bind the scene to the renderer.
    And call the render function of render_efficient_memory function.
    """

    def __init__(self, device=None):
        super(CTRenderer, self).__init__(device=device)

    def anatomy_corners_location_ndc_coord(self, ct_anatomy: AnatomyCT, camera: Camera):
        """
        Find the extent of the anatomy in x,y,z (from the cameras perspective using NDC from -1 to
        +1)
        we use anatomy_corners_location_ndc_coord in our renderer to find (min, max) width and height for raysampler.
        It helps to choose the optimal width and height for the raysampler's meshgrid and reduce time and memory.
        Returns:
            (batch_size, 8, 3)
        """
        ct_anatomy = ct_anatomy.to(self.device)
        assert ct_anatomy.get_batch_size() == camera.get_batch_size()
        batch_size = ct_anatomy.get_batch_size()
        corners_local_coord = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=DTYPE_TORCH,
            device=self.device,
        )

        corners_world_coord = ct_anatomy.local_to_world_coords(
            corners_local_coord[None].expand(batch_size, 8, 3)
        )

        # If anatomy has a orientation, we exert its orientation on camera's orientation
        self.change_camera_extrinsic_mat_based_anatomy(anatomy=ct_anatomy, camera=camera)
        corners_ndc_coord = camera.transform_points_ndc(corners_world_coord, eps=eps)
        # before return something we must camera.R and camera.T to its original values for next usages.
        self.return_back_camera_extrinsic_mat_original_value(camera=camera)
        return corners_ndc_coord

    def anatomy_corners_location_camera_coord(self, ct_anatomy: AnatomyCT, camera: Camera):
        """
        we use anatomy_corners_location_camera_coord in our renderer to find the min and max depth for raysampler.
        It helps us to choose an optimal depth for the raysampler's meshgrid and reduce requied time and memory.
        """
        ct_anatomy = ct_anatomy.to(self.device)
        assert ct_anatomy.get_batch_size() == camera.get_batch_size()
        batch_size = ct_anatomy.get_batch_size()
        # pytorch3d use this -1 and +1 as min and max depths,
        # but I think it should be changed to -1.0 + local_voxel_size_z / 2 and 1.0 - local_voxel_size_z / 2,
        # if you take a look at the creation of AbsorptionOnlyRaymarcherGeneralized you can see the point.
        corners_local_coord = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=DTYPE_TORCH,
            device=self.device,
        )

        corners_world_coord = ct_anatomy.local_to_world_coords(
            corners_local_coord[None].expand(batch_size, 8, 3)
        )

        # If anatomy has a orientation, we exert its orientation on camera's orientation
        self.change_camera_extrinsic_mat_based_anatomy(anatomy=ct_anatomy, camera=camera)
        corners_camera_coord = camera.get_world_to_view_transform().transform_points(
            corners_world_coord, eps=eps
        )
        # before return something we must camera.R and camera.T to its original values for next usages.
        self.return_back_camera_extrinsic_mat_original_value(camera=camera)
        return corners_camera_coord

    def set_raysampler(self, ct_anatomy: AnatomyCT, camera: Camera, ray_pts_num_to_grid_size_z: int = 1):
        """
        This raysampler uses a meshgrid with optimal depth, but with width and height as big as the screen
        The meshgrid has the same depth as the anatomy
        """
        # all the batches have the same grid_size, so it doesn't matter what batch index should be used.
        grid_size_z = ct_anatomy.get_grid_sizes()[
            0, 0
        ]
        corners_camera_coord = self.anatomy_corners_location_camera_coord(
            ct_anatomy=ct_anatomy, camera=camera
        )
        # should I consider off_set for min_depth and max_depth?
        min_depth = torch.min(corners_camera_coord[:, :, 2], dim=1).values
        max_depth = torch.max(corners_camera_coord[:, :, 2], dim=1).values

        raysampler = NDCGridRaysamplerGeneralized(
            image_width=self.img_pixels_num[0],
            image_height=self.img_pixels_num[1],
            n_pts_per_ray=ray_pts_num_to_grid_size_z * grid_size_z,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        return raysampler, {"min_depth": min_depth, "max_depth": max_depth}

    def set_partial_raysampler(
            self, ct_anatomy: AnatomyCT, camera: Camera, ray_pts_num_to_grid_size_z: int = 1
    ):
        """

        This raysampler uses a meshgrid with optimal width, height and depth based on the anatomy
        extends, so it doesn't meshgrid the whole space between camera and intensifier.


        Compared to set_raysampler, which grids the full image width and height, even though the
        anatomy might only cover 20 percent of it

        Args:
            ray_pts_num_to_grid_size_z: sample how often per voxel in depth (1 samples once per
            voxel in the anatomy)
        Returns:
            lowest and highest indices to render (restricted to 0 to max_img_size). If anatomy is
            off screen, we will get 0. and 1. or (max-1, max+1), which is admittedly unelegant
        """
        batchsize = ct_anatomy.get_batch_size()
        assert batchsize == camera.get_batch_size(), "Anatomies must have the same batchsize as the camera"
        # (batchsize, 8, 3) for the corners of a cube
        corners_ndc_coord = self.anatomy_corners_location_ndc_coord(ct_anatomy=ct_anatomy, camera=camera)
        min_x_anatomy_ndc_coord = torch.min(corners_ndc_coord[:, :, 0], dim=1).values
        max_x_anatomy_ndc_coord = torch.max(corners_ndc_coord[:, :, 0], dim=1).values
        min_y_anatomy_ndc_coord = torch.min(corners_ndc_coord[:, :, 1], dim=1).values
        max_y_anatomy_ndc_coord = torch.max(corners_ndc_coord[:, :, 1], dim=1).values
        assert min_x_anatomy_ndc_coord.shape[0] == batchsize, "Wrong batchsize"

        half_pix_width = 1.0 / self.img_pixels_num[0]
        half_pix_height = 1.0 / self.img_pixels_num[1]

        max_x_screen_ndc_coord = 1.0 - half_pix_width
        min_x_screen_ndc_coord = -1.0 + half_pix_width
        max_y_screen_ndc_coord = 1.0 - half_pix_height
        min_y_screen_ndc_coord = -1.0 + half_pix_height
        """
        we need to start linspace with max values since screen coord start from left to right and up to down, but
        in ndc coordinate: x: right to left, y: down to up
        """
        x_screen_grid = torch.linspace(
            max_x_screen_ndc_coord,
            min_x_screen_ndc_coord,
            self.img_pixels_num[0],
            dtype=torch.float32,
            device=self.device,
        )
        y_screen_grid = torch.linspace(
            max_y_screen_ndc_coord,
            min_y_screen_ndc_coord,
            self.img_pixels_num[1],
            dtype=torch.float32,
            device=self.device,
        )

        w_start_idx = (
            torch.argmin(
                torch.abs(max_x_anatomy_ndc_coord[:, None] - x_screen_grid.expand(batchsize, -1)), dim=1
            )
            - 1
        )
        # we are not allowed to assign w_start_idx less than zero
        w_start_idx[w_start_idx < 0] = 0
        w_end_idx = (
            torch.argmin(
                torch.abs(min_x_anatomy_ndc_coord[:, None] - x_screen_grid.expand(batchsize, -1)), dim=1
            )
            + 1
        )

        # we are not allowed to assign w_end_idx more than self.img_pixels_num[0] - 1
        w_end_idx[w_end_idx > self.img_pixels_num[0]] = self.img_pixels_num[0] - 1
        h_start_idx = (
            torch.argmin(
                torch.abs(max_y_anatomy_ndc_coord[:, None] - y_screen_grid.expand(batchsize, -1)), dim=1
            )
            - 1
        )
        # we are not allowed to assign h_start_idx less than zero
        h_start_idx[h_start_idx < 0] = 0
        h_end_idx = (
            torch.argmin(
                torch.abs(min_y_anatomy_ndc_coord[:, None] - y_screen_grid.expand(batchsize, -1)), dim=1
            )
            + 1
        )
        # we are not allowed to assign h_end_idx more than self.img_pixels_num[1] - 1
        h_end_idx[w_end_idx > self.img_pixels_num[1]] = self.img_pixels_num[1] - 1

        corners_camera_coord = self.anatomy_corners_location_camera_coord(
            ct_anatomy=ct_anatomy, camera=camera
        )
        min_depth = torch.min(corners_camera_coord[:, :, 2], dim=1).values
        max_depth = torch.max(corners_camera_coord[:, :, 2], dim=1).values
        assert w_start_idx.shape == w_end_idx.shape == h_start_idx.shape == h_end_idx.shape
        # how many voxels does the anatomy have in the depth direction
        grid_size_z = ct_anatomy.get_grid_sizes()[0, 0]
        """
        FV: not sure why we need that
        # return something camera.R and camera.T to its original values for next usages.
        camera.R = camera.tmats[:, 0:3, 0:3]
        camera.T = camera.tmats[:, 3, 0:3]
        """
        compensate = 3
        minVal = 0
        maxVal_w = int(self.img_pixels_num[0])
        maxVal_h = int(self.img_pixels_num[1])
        clamped_w_start_idx = torch.clamp(w_start_idx - compensate, min=minVal)
        clamped_w_end_idx = torch.clamp(w_end_idx + compensate, max=maxVal_w)
        clamped_h_start_idx = torch.clamp(h_start_idx - compensate, min=minVal)
        clamped_h_end_idx = torch.clamp(h_end_idx + compensate, max=maxVal_h)
        raysampler = PartialNDCGridRaysampler(
            image_width=self.img_pixels_num[0],
            image_height=self.img_pixels_num[1],
            w_start_idx=clamped_w_start_idx,
            w_end_idx=clamped_w_end_idx,
            h_start_idx=clamped_h_start_idx,
            h_end_idx=clamped_h_end_idx,
            n_pts_per_ray=ray_pts_num_to_grid_size_z * grid_size_z,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # raysampler actually modifies the indices to fit the largest one
        return raysampler, {
            "w_start_idx": raysampler.w_start_idx,
            "w_end_idx": raysampler.w_end_idx,
            "h_start_idx": raysampler.h_start_idx,
            "h_end_idx": raysampler.h_end_idx,
            "min_depth": min_depth,
            "max_depth": max_depth,
        }

    def render_efficient_memory(
            self,
            cam_index: int = 0,
            width_pixels_num: int = 512,
            height_pixels_num: int = 512,
            binary=0.15,
            scale_output=True,
            decay_type='exp',
    ):
        """just an alias for backwards compatibility"""
        return self.render(cam_index=cam_index, width_pixels_num=width_pixels_num, height_pixels_num=height_pixels_num,
                           binary=binary, scale_output=scale_output, efficient=True, decay_type=decay_type)

    def render(
            self,
            cam_index: int = 0,
            width_pixels_num: int = 512,
            height_pixels_num: int = 512,
            binary=False,
            scale_output=True,
            efficient=False,
            decay_type='exp',

    ):
        """
        It generates the rendered image of the scene from view point of the camera given by user.
        param: width_pixels_num:
        param: height_pixels_num:
        param: binary: if False image will be greyscale, if a float is given, then the images is set
            to black and white based on that threshold
        param: scale_output: if it is ture, we scale elements of the img so that the smallest element becomes 0,
               the largest becomes 1, and intermediate values are mapped to the interval [0, 1] by a linear function.
               this operation is the same operation in function plt.imshow().
               we also need to do this operation for using the function adjust_brightness_contrast.
               the pixel corresponding to air should have the maximum value (0.0).
        param: efficient: use the partial raysampler to just sample the part of the image covered by
            the anatomy instead of the full image
          ..note:: if the image has a constant value just return the image. notably important when
          we render a black image
        ..note:: replaced this legacy function with render_efficient - setting efficient=True by
        default now

        return: it returns a torch.tensor on self.device in the shape of:
                (batch_size, height_pixels_num, width_pixels_num)
        """
        if binary == True:
            raise ValueError("The interface has been changed, please pass a float treshold to binary")
        assert self.scene.anatomies_type == "CT", "Scene must only have ct anatomies"
        camera = self.scene.cameras[cam_index]
        self.set_resolution(width_pixels_num=width_pixels_num, height_pixels_num=height_pixels_num)

        raymarcher = AbsorptionOnlyRaymarcherGeneralized(decay=decay_type)
        img = torch.zeros(self.scene.get_batch_size(), *self.img_pixels_num, device=self.device)

        for anatomy in self.scene.anatomies:
            if efficient:
                raysampler, rayborder = self.set_partial_raysampler(ct_anatomy=anatomy, camera=camera)
            else:
                raysampler, rayborder = self.set_raysampler(ct_anatomy=anatomy, camera=camera)
            renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)
            self.change_camera_extrinsic_mat_based_anatomy(anatomy=anatomy, camera=camera)
            if efficient:
                img_partial = renderer(cameras=camera, volumes=anatomy)[0][..., 0]
                # add padding at the right places, not vectorized
                for i in range(img_partial.shape[0]):
                    # rayborder is shape (batchsize,)
                    padding_size = (
                        rayborder["w_start_idx"][i],
                        width_pixels_num - rayborder["w_end_idx"][i],
                        rayborder["h_start_idx"][i],
                        height_pixels_num - rayborder["h_end_idx"][i],
                    )
                    img[i] += torch.nn.functional.pad(img_partial[i], padding_size, "constant", 0)
            else:
                img = img + renderer(cameras=camera, volumes=anatomy)[0][..., 0]
            self.return_back_camera_extrinsic_mat_original_value(camera=camera)
        img_min = torch.min(img)
        img_max = torch.max(img)
        eps = 1e-10
        if not scale_output or img_max == img_min:
            rendered_img = img
        else:

            rendered_img = (img - img_min) / (img_max - img_min + eps)
            rendered_img = torch.clamp(rendered_img, min=0, max=1)

        if binary:
            rendered_img[rendered_img > binary - eps] = 1.0
            rendered_img[rendered_img < binary + eps] = 0.0
        return rendered_img
