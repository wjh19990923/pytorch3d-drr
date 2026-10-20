"""
we specify all components with respect to forward slash or backslash of the machine using "fs_" or "bs_" (when looking from the top)
the origin is at the midpoint of the bs_intensifier and the world cordinate's axes are
 - x: to the right of bs_ii (parallel to its horizontal edge)
 - y: to the top of bs_ii (parallel to its vertical edge)
 - z: from inside to outside of the bs_ii (toward its source)
the fs_ is oriented about -y

From the top, this is an approximate view:

\ bs_intensifier \                / fs_intensifier /
 \                \              /                /
  \                \            /                /
   \                \          /                /
    \                \        /                /
     \                \      /                /
      \                \    /                /
       \                \  /                /
        \                \/                /
         \               /\               /
          \             /  \             /
           \           /    \           /
            \         /      \         /
             \       /        \       /
              \     /          \     /
               \   /            \   /
                \ /   [object]   \ /
                 \                \
                / \              / \
               /   \            /   \
              /     \          /     \
             /       \        /       \
            /         \      /         \
           /           \    /           \
          /             \  /             \
         /               \/               \
        /                /\                \
       /                /  \                \
      /                /    \                \
     /                /      \                \
    /                /        \                \
   /                /          \                \
  /                /            \                \
 /                /              \                \
/    fs_source   /                \   bs_source    \


required information:
a) focal length and principal point of the cameras with respect to their corresponding intensifiers

b) center and orientation of the fs_intensifier

Not that all the dimenstions are in millimeter

For using this package you can choose the origin and orientation of the world coordinate system as you want.
But we usually define world coordinate system as the same as bs_intensifier's coordinate system.
And all the objects must be defined based on the world coordinated system.
Each object in the scene has a tmat (transfromation matrix from coordinate system of that object to world coordinate system).
so if we have a point in the coordinate system of an object and we want to represent it in world coordinate system,
we use the tmat in this manner:
point_in_world_coord = tmat(4 x 4) * point (4 x 1)
and depending on how many anatomy the tmat belongs to, it must have the form of (batch_size, 4, 4):
                                                                                            [[Rxx, Ryx, Rzx, Tx],
                                                                                            [Rxy, Ryy, Rzy, Ty],
                                                                                            [Rxz, Ryz, Rzz, Tz],
                                                                                            [ 0,   0,   0,   1]]
Camera's tmat is calculated internally.
But Anatomy's tmat can be set by user depending on its position and orientation in the scene.
for an AnatomySTL, tmat is equal to converting coordinate system of the .stl file to world coordinate system.
for an AnatomyCT, tmat is equal to converting coord inate system of the .nii file to world coordinate system.

It is possible that you created your scene what want to render a anatomy whose tmat provided by others,
and they have provided that tmat in their own coordinate system.
In this case you must know position and orienation of their world coordinate system to your world coordinate system.
and you need the transformation matrix that converts any point in their world coordinate system to you world coordinate system.
"""

import pathlib
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from stl.mesh import Mesh

from .utilities import determine_device

import importlib.util

pytorch3d_spec = importlib.util.find_spec("pytorch3d")
if pytorch3d_spec is not None:
    from pytorch3d.ops import cubify
    from pytorch3d.renderer import FoVPerspectiveCameras  # class
    from pytorch3d.renderer.mesh import TexturesVertex
    from pytorch3d.structures import Meshes as Pytorch3dMesh  # class
    from pytorch3d.structures import Volumes as Pytorch3dVolume
    # from pytorch3d.structures.volumes import VolumeLocator

    from pytorch3d.transforms import Scale, Transform3d

    print("Pytorch3d detected, importing packages.")
else:
    print("PyTorch3D not found. Skipping imports related to PyTorch3D.")


DTYPE_TORCH = torch.float32


def copy_names(names: Union[str, List[str]]):
    """
    names: it can be a string or list of strings for batched anatomy
    each time you make a copy of an anatomy, it will extract how many time the anatomy has been already copied,
    then it sum that number with one.
    return a list of number showing how many time the anatomy has been copied.
    The length of the list is equal to the batch_size
    """
    if not isinstance(names, list):
        names = [names]
    copy_numbers = []  # number of copies that names already have been
    for name in names:
        try:
            index = name.rindex("times_copy")
            try:
                copy_numbers.append(int(name[index - 1]) + 1)
            except ValueError:  # if there wasn't any number before times_copy, so that name wasn't duplicated before.
                copy_numbers.append(0)
        except ValueError:  # if it doesn't find any times_copy
            copy_numbers.append(0)

    return copy_numbers


class ModelBase:
    def __init__(self, names: Union[str, List[str]], ids: Union[None, List[None], int, List[int]] = None):
        """
        names: it can be a string or list of strings for batched anatomy
        ids: it would be needed when you are working with data base.
             otherwise, you can just ignore its value. it automatically will be handled as None.

        we create all the objects on the cpu. after creation user can change device.
        only for rendering we want user to determine device, then we change device for all objects in scene.
        """
        self.device = determine_device("cpu")
        if not isinstance(names, list):
            names = [names]
        self.names = names
        # breakpoint()
        self._N = len(names)  # I should check it if its type must be int or torch.Size()
        if not isinstance(ids, list):
            if (
                    ids is None
            ):  # this is for the moments you don't care about ids and want to create an object by default values
                ids = [None for i in range(self._N)]
            else:
                ids = [ids]
        assert len(ids) == self._N, "ids must have the same batch size as names has."
        self.ids = ids
        self.tmats = torch.eye(4).repeat(self._N, 1, 1)
        self.visibilities = torch.tensor([True]).expand(self._N)

    def set_model_matrix(self, new_tmats, is_yours: bool = True, **kwargs):
        """
        if is_yours = False: you are choosing tmat based on the data belong to others,
        and you must determine theirs_to_yours as another argument when you call this function"
        theirs_to_yours: means the transformation matrix from their world coordinate system to yours

        new_tmats and theirs_to_yours must be torch.tensor with shape of (batch_size, 4, 4) saved on cpu
        """
        self.check_validity_model_matrix(tmats=new_tmats)
        if is_yours:
            self.tmats = new_tmats.to(self.device)
        else:
            their_world_to_yours = kwargs.get("theirs_to_yours")
            if their_world_to_yours is not None:
                if their_world_to_yours.shape == torch.Size([1, 4, 4]):
                    their_world_to_yours = their_world_to_yours.expand(self._N, 4, 4)
                self.check_validity_model_matrix(tmats=their_world_to_yours)
                self.tmats = torch.matmul(their_world_to_yours, new_tmats).to(self.device)
            else:
                raise TypeError(
                    "missing 1 required argument. "
                    "if you choosed tmat based on the data belong to others, you must determine theirs_to_yours"
                )

    def apply_new_model_matrix(self, apply_tmats, do_around_com=False):
        """
        applies a new transform matrix (4x4) to the existing transform matrix.
        apply_tmats must be a torch.tensor with shape of (batch_size, 4, 4) saved on cpu
        if you want to appy a tmats around center of the mass,
        first you need translationaly transfer the anatomy so that its center is located on the origin.
        Then you can apply your tmats. After that you need to translationaly return back anatomy's center.
        """
        self.check_validity_model_matrix(tmats=apply_tmats)
        new_tmats = torch.matmul(apply_tmats, self.tmats)
        self.set_model_matrix(new_tmats=new_tmats)

    def check_validity_model_matrix(self, tmats):
        """
        every tmats must be a torch.tensor with shape of (batch_size, 4, 4) saved on cpu
        all tmats must be in form of (batch_size, 4, 4).
        for example for batch_size=1 it must be like this: [[[Rxx, Ryx, Rzx, Tx],
                                                            [Rxy, Ryy, Rzy, Ty],
                                                            [Rxz, Ryz, Rzz, Tz],
                                                            [ 0,   0,   0,   1]]]
        their determinant must equal to one.
        and their rotational part must be orthogonal
        check the datatype, the shape and the values of the transformation matrices
        """
        assert tmats.dtype == DTYPE_TORCH, "tmats must be {}".format(DTYPE_TORCH)
        assert tmats.shape == torch.Size([self._N, 4, 4]), "tmats must have the form of [{}, 4, 4]".format(self._N)
        assert torch.allclose(
            tmats[:, 3, :],
            torch.tensor([[0, 0, 0, 1]], dtype=DTYPE_TORCH, device=tmats.device).expand(self._N, 4),
        ), f"the last row of each intrinsic_affine_mat must be [0, 0, 0, 1], instead of {tmats[:, 3, :]}"

        assert torch.allclose(
            torch.det(tmats),
            torch.tensor([1.0], dtype=DTYPE_TORCH, device=tmats.device).expand(self._N),
            atol=1e-04,
        ), f"the determinant of all tmats must be equal(so close) to 1.0, instead of {torch.det(tmats)}"
        rr = torch.matmul(tmats.permute(0, 2, 1)[:, 0:3, 0:3], tmats[:, 0:3, 0:3])
        assert torch.allclose(
            rr, torch.eye(3, dtype=DTYPE_TORCH, device=tmats.device)[None].expand(self._N, 3, 3), atol=1e-05
        ), f"rotational part of all tmats must be orthogonal {rr}"

    def get_batch_size(self):
        return self._N

    def get_grid_sizes(self):
        return self._grid_sizes

    def set_visibilities(self, new_visbilities):
        # this attribute can be used for the GUI to determine which anatomy to show
        assert new_visbilities.dtype == torch.bool
        assert new_visbilities.shape == torch.Size([self._N])
        self.visibilities = new_visbilities


try:
    from pytorch3d.structures import Meshes as Pytorch3dMesh
except ImportError:
    class Pytorch3dMesh:
        """Dummy class if PyTorch3D is not available."""
        pass


class AnatomySTL(ModelBase, Pytorch3dMesh):
    # line_width is important only when you want to show just the border of rendered image.
    line_width = 3.0

    def __init__(
            self,
            names: Union[str, List[str]],
            verts: Union[torch.Tensor, List[torch.Tensor]],
            faces: Union[torch.Tensor, List[torch.Tensor]],
            verts_normals: Union[torch.Tensor, List[torch.Tensor]],
            ids: Union[None, List[None], int, List[int]] = None,

    ):
        """

        :param names:
        :param verts: It must be a torch.tensor or a list of torch.tensors.
                      the shape of each densities must be (num_verts, 3).
                      We combine them and change it into (batch_size, max_num_verts, 3)
        :param faces: It must be a torch.tensor or a list of torch.tensors.
                      the shape of each densities must be (num_faces, 3).
                      We combine them and change it into (batch_size, max_num_faces, 3)
        :param verts_normals: It must be a torch.tensor or a list of torch.tensors.
                              the shape of each densities must be (num_verts, 3).
                              We combine them and change it into (batch_size, max_num_verts, 3)
        """
        ModelBase.__init__(self, names=names, ids=ids)
        if not isinstance(verts, list):
            verts = [verts]

        if not isinstance(faces, list):
            faces = [faces]

        if not isinstance(verts_normals, list):
            verts_normals = [verts_normals]

        assert len(verts) == self._N, "verts must have the same batch size as names has."
        assert len(faces) == self._N, "faces must have the same batch size as names has."
        assert len(verts_normals) == self._N, "verts_normals must have the same batch size as names has."

        for i in range(self._N):
            assert (verts[i]).dtype == DTYPE_TORCH, "every verts must be {}".format(DTYPE_TORCH)
            assert (faces[i]).dtype == DTYPE_TORCH, "every faces must be {}".format(DTYPE_TORCH)
            assert (verts_normals[i]).dtype == DTYPE_TORCH, "every verts_normals must be {}".format(
                DTYPE_TORCH
            )

        self.textures = None
        Pytorch3dMesh.__init__(self, verts=verts, faces=faces, verts_normals=verts_normals)
        colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE_TORCH).expand(self._N, 3)
        opacities = torch.ones(self._N, dtype=DTYPE_TORCH)
        self.set_verts_features(colors, opacities)
        self.outline_modes = torch.tensor([False]).expand(self._N)

    @classmethod
    def load_data(
            cls,
            fpaths: Union[str, List[str], pathlib.Path, List[pathlib.Path]],
            ids: Union[None, List[None], int, List[int]] = None,
            multiply: int = 1
    ):
        """
        see `AnatomyCT`
        using this class method, you can create the anatomy from stl files
        """
        if not isinstance(fpaths, list):
            fpaths = [fpaths]

        for fpath in fpaths:
            assert pathlib.Path(fpath).suffix == ".stl", "fpaths must be a .stl files"

        names = []
        verts = []
        faces = []
        verts_normals = []
        for fpath in fpaths:
            mesh = Mesh.from_file(fpath)
            verts_np = mesh.vectors.reshape([-1, 3])
            faces_np = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(verts_np.shape[0] // 3)])
            faces_normals = mesh.normals[:, np.newaxis, ...]
            verts_normals_np = np.concatenate([faces_normals, faces_normals, faces_normals], axis=1).reshape(
                [-1, 3]
            )
            names.append(pathlib.Path(fpath).stem)
            verts.append(torch.from_numpy(verts_np).type(DTYPE_TORCH))
            faces.append(torch.from_numpy(faces_np).type(DTYPE_TORCH))
            verts_normals.append(torch.from_numpy(verts_normals_np).type(DTYPE_TORCH))
        if multiply > 1:
            assert len(names) == len(verts) == len(faces) == len(verts_normals) == 1, (
                "Multiply only" " works when you pass a single fpaths to load"
            )
            names = [names[0] for _ in range(multiply)]
            if ids:
                ids = [ids[0] for _ in range(multiply)]
            verts = [verts[0].view(verts[0].shape) for _ in range(multiply)]
            faces = [faces[0].view(faces[0].shape) for _ in range(multiply)]
            verts_normals = [verts_normals[0].view(verts_normals[0].shape) for _ in range(multiply)]
        return cls(names=names, verts=verts, faces=faces, verts_normals=verts_normals, ids=ids)

    def set_verts_features(self, colors, opacities):
        """
        :param colors: RGB, it must be a torch.tensor in the shape of (batch_size, 3).
                       each component of RGB must be between 0.0 and 1.0
        :param opacities: it must be a torch.tensor in the shape of (batch_size,).
                          all opacities must be between 0.0 and 1.0
        """
        from pytorch3d.renderer.mesh import TexturesVertex

        assert colors.min() >= 0.0, "colors has a value less than zero, but it shouldn't have"
        assert colors.max() <= 1.0, "colors has a value bigger than one, but it shouldn't have"
        assert colors.shape == torch.Size([self._N, 3]), "colors must have the form of [{}, 3]".format(
            self._N
        )

        assert opacities.min() >= 0.0, "alphas has a value less than zero, but it shouldn't have"
        assert opacities.max() <= 1.0, "alphas has a value bigger than one, but it shouldn't have"
        assert opacities.shape == torch.Size([self._N]), "alphas must have the form of [{}]".format(self._N)

        verts_rgb = colors.type(DTYPE_TORCH)[:, None, :].expand(self._N, self._V, 3)
        verts_opacity = opacities.type(DTYPE_TORCH)[:, None, None].expand(self._N, self._V, 1)
        verts_feature = torch.cat((verts_rgb, verts_opacity), dim=-1)
        # verts_feature = verts_rgb
        textures = TexturesVertex(verts_features=verts_feature)
        self.textures = textures
        shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
        if not shape_ok:
            msg = "Textures do not match the dimensions of Meshes."
            raise ValueError(msg)

        self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
        self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
        self.textures.valid = self.valid

    def copy(self, **kwargs):
        # make independent deep copy of an anatomy
        copy_numbers = copy_names(self.names)
        anatomy_new = AnatomySTL(
            names=[
                self.names[i] + "_{}times_copy".format(copy_numbers[i] + 1)
                for i in range(self.get_batch_size())
            ],
            verts=[verts.clone() for verts in self.verts_list()],
            faces=[faces.clone() for faces in self.faces_list()],
            verts_normals=[verts_normals.clone() for verts_normals in self.verts_normals_list()],
            ids=self.ids,
        )

        anatomy_new.set_visibilities(new_visbilities=kwargs.get("visibilities", self.visibilities.clone()))
        anatomy_new.set_model_matrix(new_tmats=kwargs.get("tmats", self.tmats.clone()))
        anatomy_new.textures = self.textures.clone()
        anatomy_new.outline_modes = self.outline_modes.clone()
        return anatomy_new

    def update_padded(self, new_verts_padded):
        """
        This is the same as the function wrote by the pytorch3d developers, but
        We need to change following command since we are inheriting our stl_anatomy from Pytorch3dMesh:
        new = self.__class__(verts=new_verts_padded, faces=self.faces_padded())
        to following command:
        new = Pytorch3dMesh(verts=new_verts_padded, faces=self.faces_padded())
        otherwise it was trying to call our __ini__ with not enough input arguments,
        and it was producing error.

        This function allows for an update of verts_padded without having to
        explicitly convert it to the list representation for heterogeneous batches.
        Returns a Meshes structure with updated padded tensors and copies of the
        auxiliary tensors at construction time.
        It updates self._verts_padded with new_verts_padded, and does a
        shallow copy of (faces_padded, faces_list, num_verts_per_mesh, num_faces_per_mesh).
        If packed representations are computed in self, they are updated as well.
        Args:
            new_verts_padded: FloatTensor of shape (N, V, 3)
        Returns:
            Meshes with updated padded representations
        """

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError("new values must have the same batch dimension.")
            if x.shape[1] != size[1]:
                raise ValueError("new values must have the same number of points.")
            if x.shape[2] != size[2]:
                raise ValueError("new values must have the same dimension.")

        check_shapes(new_verts_padded, [self._N, self._V, 3])

        new = Pytorch3dMesh(verts=new_verts_padded, faces=self.faces_padded())

        if new._N != self._N or new._V != self._V or new._F != self._F:
            raise ValueError("Inconsistent sizes after construction.")

        # overwrite the equisized flag
        new.equisized = self.equisized

        # overwrite textures if any
        new.textures = self.textures

        # copy auxiliary tensors
        copy_tensors = ["_num_verts_per_mesh", "_num_faces_per_mesh", "valid"]

        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)  # shallow copy

        # shallow copy of faces_list if any, st new.faces_list()
        # does not re-compute from _faces_padded
        new._faces_list = self._faces_list

        # update verts/faces packed if they are computed in self
        if self._verts_packed is not None:
            copy_tensors = [
                "_faces_packed",
                "_verts_packed_to_mesh_idx",
                "_faces_packed_to_mesh_idx",
                "_mesh_to_verts_packed_first_idx",
                "_mesh_to_faces_packed_first_idx",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy
            # update verts_packed
            pad_to_packed = self.verts_padded_to_packed_idx()
            new_verts_packed = new_verts_padded.reshape(-1, 3)[pad_to_packed, :]
            new._verts_packed = new_verts_packed
            new._verts_padded_to_packed_idx = pad_to_packed

        # update edges packed if they are computed in self
        if self._edges_packed is not None:
            copy_tensors = [
                "_edges_packed",
                "_edges_packed_to_mesh_idx",
                "_mesh_to_edges_packed_first_idx",
                "_faces_packed_to_edges_packed",
                "_num_edges_per_mesh",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy

        # update laplacian if it is compute in self
        if self._laplacian_packed is not None:
            new._laplacian_packed = self._laplacian_packed

        assert new._verts_list is None
        assert new._verts_normals_packed is None
        assert new._faces_normals_packed is None
        assert new._faces_areas_packed is None

        return new

    def to(self, new_device: Union[str, torch.device]):
        """
        transfer all torch.tensor attributes to new_device
        it changes self and returns self.
        if new_device is the same with old_device, it does nothing
        """
        old_device = self.device
        new_device = determine_device(new_device)
        if not new_device == old_device:
            self.device = new_device
            self.tmats = self.tmats.to(new_device)
            self._verts_list = [v.to(new_device) for v in self._verts_list]
            self._faces_list = [f.to(new_device) for f in self._faces_list]

            for k in self._INTERNAL_TENSORS:
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(self, k, v.to(new_device))

            self.textures = self.textures.to(new_device)

        return self


def join_anatomies_stl_as_batch(anatomies: List[AnatomySTL]):
    assert isinstance(anatomies, list), "anatomies must be inside of a list"

    anatomy_new = AnatomySTL(
        names=[name for anatomy in anatomies for name in anatomy.names],
        verts=[v for anatomy in anatomies for v in anatomy.verts_list()],
        faces=[f for anatomy in anatomies for f in anatomy.faces_list()],
        verts_normals=[v for anatomy in anatomies for v in anatomy.verts_normals_list()],
        ids=[id for anatomy in anatomies for id in anatomy.ids],
    )

    # We don't combine line_width. Since if anatomies have different line_width we don't know what we should do.
    # So after batching, the new anatomy will get the default value for line_width.
    # And the user can change it if he/she wants to.
    all_textures = [anatomy.textures for anatomy in anatomies]
    first_texture = all_textures[0]
    anatomy_new.textures = first_texture.join_batch(all_textures[1:])
    anatomy_new.visibilities = torch.cat([anatomy.visibilities for anatomy in anatomies], dim=0)
    anatomy_new.outline_modes = torch.cat([anatomy.outline_modes for anatomy in anatomies], dim=0)
    anatomy_new.tmats = torch.cat([anatomy.tmats for anatomy in anatomies], dim=0)
    return anatomy_new


try:
    from pytorch3d.structures import Volumes as Pytorch3dVolume
except ImportError:
    class Pytorch3dVolume:
        """Dummy class if PyTorch3D is not available."""
        pass


class AnatomyCT(ModelBase, Pytorch3dVolume):
    def __init__(
            self,
            names: Union[str, List[str]],
            densities: Union[torch.Tensor, List[torch.Tensor]],
            intrinsic_affine_mats: Union[torch.Tensor, List[torch.Tensor]],
            ids: Union[None, List[None], int, List[int]] = None,
    ):
        """
        :param names:
        :param densities: It must be a torch.tensor or a list of torch.tensors
            the shape of each densities must be (1, 1, depth, height, width).
            We combine them and change it into (batch_size, 1, depth, height, width)
        :param intrinsic_affine_mats: It must be a torch.tensor or a list of torch.tensors
            each matrix (which is like to tmat) transforms voxel coordinate (indices) to the
            anatomy's local coordinate (x, y ,z).
           ..warning: intrinsic_affine_mats differs from anatomy's tmat
           anatomy's tmat determines the pose of the anatomy in the scene,
           so it is a transformation from the anatomy's local coordinate
           to scene's world coordinate
           we use the intrinsic_affine_mat in this manner:
           pose_in_that_anatomy_local_coord = intrinsic_affine_mat(1, 4 x 4) * voxel_indix (4 x 1).
           each intrinsic_affine_mat must be in the form of (1, 4, 4).
           We combine them and change it into (batch_size, 4, 4)

        In the __init__, we reduce the grid_size of anatomy by eliminating zero-value voxels to reduce required memory for rendering.
        So we store intrinsic_affine_mats given by user as the _original_intrinsic_affine_mat.
        After reducing the grid_size, we change the _original_intrinsic_affine_mat and store it as _intrinsic_affine_mat
        If you want to know the local location of a voxel, you can use one of those affine_mat depending on
        whether the voxel indices have been chosen after or before creating your ct_anatomy.

        Note that the densities are in order depth,height,width, but
        when you want to multiply a voxel index by intrinsic_affine_mat, you must define your voxel index as following:
        voxel_index = torch.tensor([[width_index],[height_index],[depth_index],[1]]).type_as(intrinsic_affine_mat)

        Note that list and torch.tensor are mutable, so
        so if you want to keep your densities and intrinsic_affine_mat unchanged,
        you must clone them when you want to pass them as input arguments

        .. warning:: give all parameters on "cpu", and only later transfer using self.to()
        """
        ModelBase.__init__(self, names=names, ids=ids)
        densities_t = densities
        if isinstance(densities, list):
            densities_t = torch.cat(densities, dim=0)
        Pytorch3dVolume.__init__(self, densities=densities_t)
        if not isinstance(densities, list):
            densities = [densities]
        if not isinstance(intrinsic_affine_mats, list):
            intrinsic_affine_mats = [intrinsic_affine_mats]
        assert len(densities) == self._N, "densities must have the same batch size as names has."
        assert (
            len(intrinsic_affine_mats) == self._N
        ), "intrinsic_affine_mat must have the same batch size as names has."

        self._original_intrinsic_affine_mat = torch.cat(intrinsic_affine_mats, dim=0)
        _, self._original_grid_sizes = self._convert_densities_features_to_tensor(
            [densities[i][0, ...] for i in range(self._N)], "densities"
        )
        self._set_original_local_to_world_transform(intrinsic_affine_mat=self._original_intrinsic_affine_mat)

        diag_elem = torch.eye(3, dtype=bool)[None]
        transforms_sequence = []
        for i in range(self._N):
            assert (densities[i]).dtype == DTYPE_TORCH, "each densities must be {}".format(DTYPE_TORCH)
            assert (
                densities[i].ndim == 5
            ), "each densities must have the shape of [1, 1, depth, height, width]"
            assert (
                intrinsic_affine_mats[i]
            ).dtype == DTYPE_TORCH, "each intrinsic_affine_mat must be {}".format(DTYPE_TORCH)
            assert intrinsic_affine_mats[i].shape == torch.Size(
                [1, 4, 4]
            ), "each intrinsic_affine_mat must have the shape of [1, 4, 4]"
            assert torch.allclose(
                intrinsic_affine_mats[i][:, 3, :],
                torch.tensor([[0, 0, 0, 1]], dtype=DTYPE_TORCH, device=intrinsic_affine_mats[i].device),
            ), "the last row of each intrinsic_affine_mat must be [0, 0, 0, 1]"
            temp = torch.matmul(
                intrinsic_affine_mats[i][:, 0:3, 0:3], intrinsic_affine_mats[i].permute(0, 2, 1)[:, 0:3, 0:3]
            )
            if torch.allclose(
                    temp[~diag_elem],
                    torch.zeros(6, dtype=DTYPE_TORCH, device=intrinsic_affine_mats[i].device),
                    atol=1e-06,
            ):
                # scale * rotation = intrinsic_affine_mat[:, 0:3, 0:3]
                transforms_sequence.append("rst")  # first rotation, then scale and then translation.
            else:
                temp = torch.matmul(
                    intrinsic_affine_mats[i].permute(0, 2, 1)[:, 0:3, 0:3],
                    intrinsic_affine_mats[i][:, 0:3, 0:3],
                )
                if torch.allclose(
                        temp[~diag_elem],
                        torch.zeros(6, dtype=DTYPE_TORCH, device=intrinsic_affine_mats[i].device),
                        atol=1e-06,
                ):
                    # rotation * scale = intrinsic_affine_mat[:, 0:3, 0:3]
                    transforms_sequence.append("srt")  # first scale, then rotation and then translation.
                else:
                    raise ValueError(
                        "each (intrinsic_affine_mat * intrinsic_affine_mat') or "
                        "(intrinsic_affine_mat' * intrinsic_affine_mat) must be diagonal"
                    )
            # note list is mutable object, so after this line, the inputs of the object creation will be changed too
            densities[i], intrinsic_affine_mats[i] = self.eliminate_extra_zero_voxels(
                densities=densities[i], intrinsic_affine_mat=intrinsic_affine_mats[i]
            )
        """
        each densities may have different densities size.
        so for combining them into one tensor, we should first make them to have the same sizes.
        """
        max_size = self.find_max_size_of_multiple_densities(densities)  # z, y ,x
        for i in range(self._N):
            grid_size = densities[i].shape[2:]
            pad_size_depth_far = int(max_size[0] - grid_size[0])
            pad_size_y_down = int(max_size[1] - grid_size[1])
            pad_size_width_right = int(max_size[2] - grid_size[2])
            padding_size = (0, pad_size_depth_far, 0, pad_size_y_down, 0, pad_size_width_right)
            densities[i], intrinsic_affine_mats[i] = self.padding_with_zero_voxels(
                densities=densities[i],
                intrinsic_affine_mat=intrinsic_affine_mats[i],
                padding_size=padding_size,
            )
        densities = torch.cat(densities, dim=0)
        intrinsic_affine_mat = torch.cat(intrinsic_affine_mats, dim=0)
        assert (
            densities.device == intrinsic_affine_mat.device
        ), "densities and intrinsic_affine_mats must have the same devices"
        densities, grid_sizes = self._convert_densities_features_to_tensor(densities, "densities")

        # take device from densities
        self.device = densities.device
        self._intrinsic_affine_mat = intrinsic_affine_mat
        self._transforms_sequence = transforms_sequence
        # assign to the internal buffers
        self._densities = densities
        self._grid_sizes = grid_sizes

        # handle features
        self._features = None

        # set the local_to_world transform
        self._set_local_to_world_transform(intrinsic_affine_mat=self._intrinsic_affine_mat)
        local_to_world_transformers = self.determine_voxel_size_translation_scale_rotation()
        self._voxel_size = local_to_world_transformers[0]
        self._volume_translation = local_to_world_transformers[1]
        self._volume_scale = local_to_world_transformers[2]
        self._volume_rotation = local_to_world_transformers[3]
        # print(self._grid_sizes)
        # assign a coordinate transformation member
        # print(self._voxel_size)
        # self.locator = VolumeLocator(
        #     batch_size=self._densities[0].shape[0],
        #     grid_sizes=self._grid_sizes,
        #     voxel_size=1.0,
        #     volume_translation=self._volume_translation[:,:3,3],
        #     device=self.device,
        # )

    #
    def get_local_to_world_coords_transform(self) -> "Transform3d":
        """
        Return a Transform3d object that converts points in the
        the local coordinate frame of the volume to world coordinates.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **local_to_world_transform**: A Transform3d object converting
                points from local coordinates to the world coordinates.
        """
        from pytorch3d.transforms import Transform3d

        return self._local_to_world_transform

    def get_world_to_local_coords_transform(self) -> "Transform3d":
        """
        Return a Transform3d object that converts points in the
        world coordinates to the local coordinate frame of the volume.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **world_to_local_transform**: A Transform3d object converting
                points from world coordinates to local coordinates.
        """
        from pytorch3d.transforms import Transform3d

        return self.get_local_to_world_coords_transform().inverse()

    def local_to_world_coords(self, points_3d_local: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_local` of shape
        (minibatch, ..., dim) in the local coordinate frame of the volume
        to the world coordinates.

        Args:
            **points_3d_local**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_world**: `points_3d_local` converted to the world
                coordinates of the volume of shape `(minibatch, ..., 3)`.
        """
        pts_shape = points_3d_local.shape
        return self.get_local_to_world_coords_transform().transform_points(
            points_3d_local.view(pts_shape[0], -1, 3)).view(pts_shape)

    def world_to_local_coords(self, points_3d_world: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_world` of shape
        (minibatch, ..., dim) in the world coordinates to
        the local coordinate frame of the volume. Local volume
        coordinates are scaled s.t. the coordinates along one side of the volume
        are in range [-1, 1].

        Args:
            **points_3d_world**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_local**: `points_3d_world` converted to the local
                volume coordinates of shape `(minibatch, ..., 3)`.
        """
        pts_shape = points_3d_world.shape
        return self.get_world_to_local_coords_transform().transform_points(
            points_3d_world.view(pts_shape[0], -1, 3)).view(pts_shape)

    @classmethod
    def load_data(
            cls,
            fpaths: Union[str, List[str], pathlib.Path, List[pathlib.Path]],
            ids: Union[None, List[None], int, List[int]] = None,
            multiply=1,
    ):
        """
        This class method helps you to create your anatomy from nii files.
        Args:
          fpaths: a filepath or iterable of a filepath
          ids: anatomy-id or iterable of anatomy-ids
          multiply: if you pass one anatomy, you can create a batch
          ..note:: the difference to just passing n times the same fpath is that you would
            need n times the memory and have n times the loadtime, this way the memory intensive
            mesh is just a view
        """

        if not isinstance(fpaths, list):
            fpaths = [fpaths]

        for fpath in fpaths:
            assert (
                pathlib.Path(fpath).suffix == ".nii" or pathlib.Path(fpath).suffix == ".dicom"
            ), "fpaths must be a .nii or .dicom files"

        names = []
        densities = []
        intrinsic_affine_mats = []
        for i, fpath in enumerate(fpaths):
            if pathlib.Path(fpath).suffix == ".nii":
                names.append(pathlib.Path(fpath).stem)
                ct_info = nib.load(pathlib.Path(fpath))
                hounsfield_values = np.asanyarray(ct_info.dataobj)
                hounsfield_values = torch.from_numpy(hounsfield_values).type(DTYPE_TORCH)
                # densities must has the shape of (depth, height, width) not (width, height, depth)
                hounsfield_values = hounsfield_values.permute(2, 1, 0)
                hounsfield_min = torch.abs(torch.min(hounsfield_values))
                assert not float(
                    hounsfield_min) == 0., "Found 0 as minimum value, are you sure you are using Hounsfield units?"
                lin_attenuation_coeff = 1.0 + hounsfield_values / hounsfield_min
                densities.append(lin_attenuation_coeff[None, None])
                volume_size_unit = ct_info.header.get_xyzt_units()[0]
                assert volume_size_unit == "mm", (
                    "Note nii file should has a size unit in mm. If doesn't, "
                    "correct intrinsic_matrix by proper coefficient to convert it in mm"
                )
                ct_affine = torch.from_numpy(ct_info.affine).type(DTYPE_TORCH)[None]
                if torch.allclose(
                        ct_affine[:, 3, :],
                        torch.tensor([[0, 0, 0, 1]], dtype=DTYPE_TORCH, device=ct_affine.device),
                ):
                    intrinsic_affine_mats.append(ct_affine)
                else:
                    intrinsic_affine_mats.append(ct_affine.permute(0, 2, 1))
            elif pathlib.Path(fpath).suffix == ".dicom":
                raise NotImplemented
        if multiply > 1:
            assert len(names) == len(densities) == len(intrinsic_affine_mats) == 1, (
                "Multiply only" " works when you pass a single fpaths to load"
            )
            names = [names[0] for _ in range(multiply)]
            if ids:
                ids = [ids[0] for _ in range(multiply)]
            densities = [densities[0].view(densities[0].shape) for _ in range(multiply)]
            intrinsic_affine_mats = [
                intrinsic_affine_mats[0].clone() for _ in range(multiply)
            ]
        return cls(names=names, densities=densities, intrinsic_affine_mats=intrinsic_affine_mats, ids=ids)

    def _set_local_to_world_transform(self, intrinsic_affine_mat):
        """
        pytorch3d has a specific coordinate system for ct anatomy.
        The center of the voxel indexed with `[·, ·, 0, 0, 0]` in the volume
        has its 3D local coordinate set to `[-1, -1, -1]`, while the voxel
        at index `[·, ·, depth_i-1, height_i-1, width_i-1]` has its
        3D local coordinate set to `[1, 1, 1]`

        This function find the transformation matrix for converting the coordinate of a voxel between [-1, -1, -1] and
        [1, 1, 1] to its coordinate in the world.
        Note that _local_to_world_transform.get_matrix() seems like transposed tmats
        """
        from pytorch3d.transforms import Scale, Transform3d

        volume_size_zyx = self._grid_sizes.to(DTYPE_TORCH)
        # print(volume_size_zyx)
        volume_size_xyz = volume_size_zyx[:, [2, 1, 0]]
        # note that transformers in pytorch3d is transposed of our transformers
        world_to_voxel_index = Transform3d(matrix=torch.linalg.inv(intrinsic_affine_mat).permute(0, 2, 1))
        voxel_indix_to_local = Scale(2.0 / (volume_size_xyz - 1), device=self.device).translate(
            -torch.ones(self._N, 3, dtype=DTYPE_TORCH)
        )
        self._local_to_world_transform = world_to_voxel_index.compose(voxel_indix_to_local).inverse()

    def _set_original_local_to_world_transform(self, intrinsic_affine_mat):
        from pytorch3d.transforms import Scale, Transform3d

        volume_size_zyx = self.original_grid_sizes().to(DTYPE_TORCH)
        volume_size_xyz = volume_size_zyx[:, [2, 1, 0]]
        # note that transformers in pytorch3d is transposed of our transformers
        world_to_voxel_index = Transform3d(matrix=torch.linalg.inv(intrinsic_affine_mat).permute(0, 2, 1))
        voxel_indix_to_local = Scale(2.0 / (volume_size_xyz - 1), device=self.device).translate(
            -torch.ones(self._N, 3, dtype=DTYPE_TORCH)
        )
        self._original_local_to_world_transform = world_to_voxel_index.compose(voxel_indix_to_local).inverse()

    def determine_voxel_size_translation_scale_rotation(self):
        """
        This function uses grid_size and intrinsic_affine_mat to determine ...
        scaling, rotational and translational part of intrinsic_affine_mat.
        diagonal elements of intrinsic_affine_mat can be negative numbers,
        we consider voxel_size and volume_scale as positive numbers, so
        volume_rotation takes responsibility of the sign of diagonal elements.
        """
        volume_size_zyx = self._grid_sizes.to(DTYPE_TORCH)
        volume_size_xyz = volume_size_zyx[:, [2, 1, 0]]
        # we need to transpose for returning back transformer from pytorch3d format into our format
        local_to_world_mat = self._local_to_world_transform.get_matrix().permute(0, 2, 1)
        volume_translation = torch.eye(4, dtype=DTYPE_TORCH)[None].expand(self._N, 4, 4).clone()
        volume_scale = torch.eye(4, dtype=DTYPE_TORCH, device=volume_size_zyx.device)[None].expand(self._N, 4,
                                                                                                   4).clone()
        volume_rotation = torch.eye(4, dtype=DTYPE_TORCH)[None].expand(self._N, 4, 4).clone()
        volume_translation[:, 0:3, 3] = local_to_world_mat[:, 0:3, 3]
        self._intrinsic_affine_mat = self._intrinsic_affine_mat.to(device=volume_size_zyx.device)
        for i in range(self._N):
            if self._transforms_sequence[i] == "rst":  # first rotation, then scale and then translation.
                # scale * rotation = intrinsic_affine_mat[:, 0:3, 0:3]
                temp = torch.matmul(
                    self._intrinsic_affine_mat[i, 0:3, 0:3],
                    self._intrinsic_affine_mat[i, 0:3, 0:3].permute(1, 0),
                )
                volume_scale[i, 0:3, 0:3] = torch.sqrt(temp)
                volume_rotation[i, 0:3, 0:3] = torch.matmul(
                    torch.linalg.inv(volume_scale[i, 0:3, 0:3]).to(self.device),
                    self._intrinsic_affine_mat[i, 0:3, 0:3]
                )
            elif self._transforms_sequence[i] == "srt":  # first scale, then rotation and then translation.
                # rotation * scale = intrinsic_affine_mat[:, 0:3, 0:3]
                temp = torch.matmul(
                    self._intrinsic_affine_mat[i, 0:3, 0:3].permute(1, 0),
                    self._intrinsic_affine_mat[i, 0:3, 0:3],
                )
                volume_scale[i, 0:3, 0:3] = torch.sqrt(temp)
                volume_rotation[i, 0:3, 0:3] = torch.matmul(
                    self._intrinsic_affine_mat[i, 0:3, 0:3], torch.linalg.inv(volume_scale[i, 0:3, 0:3])
                )
        voxel_size = torch.vstack(
            [
                2.0 * volume_scale[i, 0:3, 0:3].diagonal() / (volume_size_xyz[i, 0:3] - 1)
                for i in range(self._N)
            ]
        )
        return voxel_size, volume_translation, volume_scale, volume_rotation

    def original_intrinsic_affine_mat(self):
        return self._original_intrinsic_affine_mat

    def original_grid_sizes(self):
        return self._original_grid_sizes

    def intrinsic_affine_mat(self):
        return self._intrinsic_affine_mat

    def voxel_size(self):
        return self._voxel_size

    def volume_translation(self):
        return self._volume_translation

    def volume_scale(self):
        return self._volume_scale

    def volume_rotation(self):
        return self._volume_rotation

    def transforms_sequence(self):
        return self._transforms_sequence

    @staticmethod
    def eliminate_extra_zero_voxels(
            densities,
            intrinsic_affine_mat,
            eliminating_size: Union[None, Tuple[int, int, int, int, int, int]] = None,
            threshold: float = 0.1,
    ):
        """
        In order to reduce required memory and time for rendering, we eliminate zero-value voxels from anatomy.
        For doing that we also need to change intrinsic_affine_mat since grid_size is not the same as it was.

        :param densities: it must be in the from of (batch_size, 1, depth, height, width)
        :param intrinsic_affine_mat: it must be in the from of (batch_size, 4, 4)
        :param eliminating_size: it must be None or as the form of:
                                 (pad_size_depth_near(lowest_depth_index), pad_size_depth_far((highest_depth_index)),
                                  pad_size_height_up(lowest_height_index), pad_size_y_down(highest_height_index),
                                  pad_size_width_left(lowest_width_index), pad_size_width_right(highest_width_index))
        :param threshold: Voxels have the value less than threshold are considered as zero voxels.
                        When eliminating_size is not None, the threshold is not considered any more.
        :return: new densities and new affine_mat in the shape of (batch_size, 4, 4)
        """

        def find_first_and_last_non_zero_voxel(densities, threshold):
            """

            :param threshold: Voxels have the value less than threshold are considered as zero voxels
            :return:
            """
            # In first or last non zero index the order is depth(z), height(y) and width(x)
            first_non_zeros = torch.Tensor().to(densities.device)
            last_non_zeros = torch.Tensor().to(densities.device)
            for i in range(densities.shape[0]):
                non_zero_voxels = torch.nonzero(densities[i, 0, ...] > threshold)
                first_non_zeros = torch.cat(
                    (
                        first_non_zeros,
                        torch.tensor(
                            [
                                non_zero_voxels[:, 0].min(),
                                non_zero_voxels[:, 1].min(),
                                non_zero_voxels[:, 2].min(),
                            ],
                            device=non_zero_voxels.device,
                        )[None],
                    ),
                    dim=0,
                )
                last_non_zeros = torch.cat(
                    (
                        last_non_zeros,
                        torch.tensor(
                            [
                                non_zero_voxels[:, 0].max(),
                                non_zero_voxels[:, 1].max(),
                                non_zero_voxels[:, 2].max(),
                            ],
                            device=non_zero_voxels.device,
                        )[None],
                    ),
                    dim=0,
                )
            first_non_zero = torch.tensor(
                [first_non_zeros[:, 0].min(), first_non_zeros[:, 1].min(), first_non_zeros[:, 2].min()],
                device=first_non_zeros.device,
            ).to(dtype=torch.int16)
            last_non_zero = torch.tensor(
                [last_non_zeros[:, 0].max(), last_non_zeros[:, 1].max(), last_non_zeros[:, 2].max()],
                device=last_non_zeros.device,
            ).to(dtype=torch.int16)
            return first_non_zero.tolist(), last_non_zero.tolist()

        # index 1, 2 and 3 are corresponding to z, y, x
        assert densities.ndim == 5, "densities must only have 5 dimension"
        assert intrinsic_affine_mat.ndim == 3, "intrinsic_affine_mat must only have 3 dimension"
        _, _, de, he, wi = densities.shape

        if eliminating_size is None:
            first_non_zero, last_non_zero = find_first_and_last_non_zero_voxel(
                densities=densities, threshold=threshold
            )
            eliminating_size = (
                first_non_zero[0],
                (de - 1) - last_non_zero[0],
                first_non_zero[1],
                (he - 1) - last_non_zero[1],
                first_non_zero[2],
                (wi - 1) - last_non_zero[2],
            )

        new_densities = densities[
            :,
            :,
            eliminating_size[0]: de - eliminating_size[1],
            eliminating_size[2]: he - eliminating_size[3],
            eliminating_size[4]: wi - eliminating_size[5],
        ].clone()

        new_intrinsic_affine_mat = intrinsic_affine_mat.clone()
        shift = torch.matmul(
            intrinsic_affine_mat[:, 0:3, 0:3],
            torch.tensor(
                [[eliminating_size[4]],
                 [eliminating_size[2]],
                 [eliminating_size[0]]],
                dtype=DTYPE_TORCH,
                device=intrinsic_affine_mat.device),
        )

        new_intrinsic_affine_mat[:, 0:3, 3] = intrinsic_affine_mat[:, 0:3, 3] + shift[:, 0:3, 0]
        return new_densities, new_intrinsic_affine_mat

    @staticmethod
    def padding_with_zero_voxels(
            densities,
            intrinsic_affine_mat,
            padding_size: Tuple[int, int, int, int, int, int],
            padding_value: float = 0,
    ):
        """
        In order to batch several anatomies with different grid_size, we need to pad them to make their size equal.
        For doing that we also also need to chagne intrinsic_affine_mat since grid_size is not the same as it was.

        :param densities: it must be in the from of (batch_size, 1, depth, height, width)
        :param intrinsic_affine_mat: it must be in the from of (batch_size, 4, 4)
        :param padding_size: it must be as the form of:
                             (pad_size_depth_near(lowest_depth_index), pad_size_depth_far((highest_depth_index)),
                              pad_size_height_up(lowest_height_index), pad_size_y_down(highest_height_index),
                              pad_size_width_left(lowest_width_index), pad_size_width_right(highest_width_index))
        :param padding_value:
        :return: new densities and new affine_mat in the shape of (batch_size, 4, 4)
        """
        assert densities.ndim == 5, "densities must only have 5 dimension"
        assert intrinsic_affine_mat.ndim == 3, "intrinsic_affine_mat must only have 3 dimension"
        # FV: dont trust this otherwise
        densities = densities.clone()
        intrinsic_affine_mat.clone()
        # We need to change order of padding size
        padding_size_changed_order = []
        padding_size_changed_order.append(padding_size[4]), padding_size_changed_order.append(padding_size[5])
        padding_size_changed_order.append(padding_size[2]), padding_size_changed_order.append(padding_size[3])
        padding_size_changed_order.append(padding_size[0]), padding_size_changed_order.append(padding_size[1])
        new_densities = torch.nn.functional.pad(
            densities, padding_size_changed_order, "constant", padding_value
        )

        new_intrinsic_affine_mat = intrinsic_affine_mat.clone()

        shift = torch.matmul(
            intrinsic_affine_mat[:, 0:3, 0:3],
            torch.tensor(
                [[padding_size[4]], [padding_size[2]], [padding_size[0]]],
                dtype=DTYPE_TORCH,
                device=intrinsic_affine_mat.device,
            ),
        )
        new_intrinsic_affine_mat[:, 0:3, 3] = intrinsic_affine_mat[:, 0:3, 3] - shift[:, 0:3, 0]
        return new_densities, new_intrinsic_affine_mat

    @staticmethod
    def find_max_size_of_multiple_densities(list_of_densities: List[torch.Tensor]):
        """
        :param list_of_densities: each densities of the list must be in form of (batch_size, 1, depth, height, width)
        """
        assert isinstance(list_of_densities, list)
        for each_densities in list_of_densities:
            assert each_densities.ndim == 5, "each densities must exactly have 5 dimension"
        max_grid_size_x = 1
        max_grid_size_y = 1
        max_grid_size_z = 1
        for each_densities in list_of_densities:
            grid_size = each_densities.shape[
                2:
            ]  # first index belongs to batch_size and second is alwasy equalt to 1.
            if grid_size[0] > max_grid_size_z:
                max_grid_size_z = grid_size[0]

            if grid_size[1] > max_grid_size_y:
                max_grid_size_y = grid_size[1]

            if grid_size[2] > max_grid_size_x:
                max_grid_size_x = grid_size[2]

        return max_grid_size_z, max_grid_size_y, max_grid_size_x

    def copy(self, **kwargs):
        # make a deep copy independent of previous anatomy
        copy_numbers = copy_names(self.names)

        anatomy_new = AnatomyCT(
            names=[
                self.names[i] + "_{}times_copy".format(copy_numbers[i] + 1)
                for i in range(self.get_batch_size())
            ],
            densities=self.densities().clone(),
            intrinsic_affine_mats=self.intrinsic_affine_mat().clone(),
            ids=self.ids,
        )

        anatomy_new.set_visibilities(new_visbilities=kwargs.get("visibilities", self.visibilities.clone()))
        anatomy_new.set_model_matrix(new_tmats=kwargs.get("tmats", self.tmats.clone()))
        return anatomy_new

    def to(self, new_device: Union[str, torch.device]):
        """
        transfer all torch.tensor attributes to new_device
        it changes self and returns self.
        if new_device is the same with old_device, it does nothing
        """
        old_device = self.device
        new_device = determine_device(new_device)

        if not new_device == old_device:
            for k in dir(self):
                v = getattr(self, k)
                if k == "device":
                    setattr(self, k, new_device)
                if torch.is_tensor(v):
                    setattr(self, k, v.to(new_device))

            self._local_to_world_transform = self._local_to_world_transform.to(new_device)
            self._original_local_to_world_transform = self._original_local_to_world_transform.to(new_device)
        return self


def join_anatomies_ct_as_batch(anatomies: List[AnatomyCT]):
    assert isinstance(anatomies, list), "anatomies must be inside of a list"

    anatomy_new = AnatomyCT(
        names=[name for anatomy in anatomies for name in anatomy.names],
        densities=[
            anatomy.densities()[i, ...][None]
            for anatomy in anatomies
            for i in range(anatomy.get_batch_size())
        ],
        intrinsic_affine_mats=[
            anatomy.intrinsic_affine_mat[i, ...][None]
            for anatomy in anatomies
            for i in range(anatomy.get_batch_size())
        ],
        ids=[id for anatomy in anatomies for id in anatomy.ids],
    )

    anatomy_new.visibilities = torch.cat([anatomy.visibilities for anatomy in anatomies], dim=0)
    anatomy_new.tmats = torch.cat([anatomy.tmats for anatomy in anatomies], dim=0)
    return anatomy_new


def anatomy_ct_to_anatomy_stl(anatomy_ct: AnatomyCT, thresh=0.1):
    if torch.cuda.is_available():
        device = "cuda"
        mesh_pytorch3d = cubify(anatomy_ct.densities()[:, 0, ...].to(device), thresh=0.1, align="center")
        # we return back the mesh to cpu, since our calculation is based on the cpu.
        # only for rendering we want user to determine device, then we change device for all objects in scene.
        mesh_pytorch3d = mesh_pytorch3d.to(device)
    else:
        # Be careful if cuda is not available, cubify is not trustable.
        # You can take a look at this issue:
        mesh_pytorch3d = cubify(anatomy_ct.densities()[:, 0, ...], thresh=0.1, align="center")

    verts_local = mesh_pytorch3d.verts_list()
    faces = mesh_pytorch3d.faces_list()
    meshes = []
    for i, face in enumerate(faces):
        ct_temp = AnatomyCT(
            "temporary_ct",
            anatomy_ct.densities()[i, ...][None],
            anatomy_ct.intrinsic_affine_mat()[i, ...][None],
        )
        verts_world = ct_temp.local_to_world_coords(verts_local[i][None])
        verts_redundant = torch.Tensor().type(DTYPE_TORCH).to(device=mesh_pytorch3d.device)
        verts_normals_redundant = torch.Tensor().type(DTYPE_TORCH).to(device=mesh_pytorch3d.device)
        for i in range(face.shape[0]):
            first_p = verts_world[0, face[i, 0], :]
            second_p = verts_world[0, face[i, 1], :]
            third_p = verts_world[0, face[i, 2], :]
            face_normal = torch.nn.functional.normalize(
                torch.cross(second_p - first_p, third_p - first_p, dim=0), dim=0
            )
            verts_redundant = torch.cat((verts_redundant, torch.vstack((first_p, second_p, third_p))), dim=0)
            verts_normals_redundant = torch.cat(
                (verts_normals_redundant, torch.vstack((face_normal, face_normal, face_normal))), dim=0
            )

        faces_redundant = torch.tensor(
            [
                [i * 3, i * 3 + 1, i * 3 + 2]
                for i in range(torch.div(verts_redundant.shape[0], 3, rounding_mode="floor"))
            ],
            dtype=DTYPE_TORCH,
            device=mesh_pytorch3d.device,
        )
        meshes.append(
            {"verts": verts_redundant, "faces": faces_redundant, "verts_normals": verts_normals_redundant}
        )
    surface_anatomy = AnatomySTL(
        names=[anatomy_ct.names[i] + "_surface" for i in range(anatomy_ct.get_batch_size())],
        verts=[meshes[i]["verts"] for i in range(anatomy_ct.get_batch_size())],
        faces=[meshes[i]["faces"] for i in range(anatomy_ct.get_batch_size())],
        verts_normals=[meshes[i]["verts_normals"] for i in range(anatomy_ct.get_batch_size())],
    )
    surface_anatomy.set_model_matrix(anatomy_ct.tmats)
    return surface_anatomy


try:
    from pytorch3d.renderer import FoVPerspectiveCameras
except ImportError:
    class FoVPerspectiveCameras:
        """Dummy class if PyTorch3D is not available."""
        pass


class Camera(FoVPerspectiveCameras):
    """
    Each Scene can have multiples cameras.
    Each camera is defined based on its calibration parameters and intensifier.
    For creating a camera, we need to define the intensifier by determining the location of its center
    and the direction of its normal vector and vertical vector in the world coordinate system.
    The normal vector of intensifier is the vector perpendicular to intensifier and toward the camera.
    The vertical vector of intensifier is parallel to the left and right edges of intensifier.
    We also need to determine the location of the intensifier's center w.r.t world coordinate system.
    Then we need to determine the length of intensifier's edges in millimeter.
    The horizontal and vertical value for determining the distance of principal point to intensifier's center must be set w.r.t intensifier's coord system.
    The principal point is the projection of the source (pinhole) on the intensifier.
    principal_points_v must be determined in direction of the intensifier's vertical vector.
    It can be negative or positive.
    principal_points_h must be determined in direction of the intensifier's horizontal vector.
    It can be negative or positive.
    You can determine the intensifier's horizontal vector by cross(screen_vertical, screen_normal).
    It means, horizontal_vector, vertical_vector and normal_vector are corresponding to x, y and z axes of intensifier.
    focal length is vertical distance of the source to the intensifier's plane, and it can be only positive.
    """

    def __init__(
            self,
            names: Union[str, List[str]],
            screen_center_poses: Union[Tuple[float, float, float], torch.Tensor],
            screen_normals: Union[Tuple[float, float, float], torch.Tensor],
            screen_verticals: Union[Tuple[float, float, float], torch.Tensor],
            screen_sizes_h: Union[float, torch.Tensor],
            screen_sizes_v: Union[float, torch.Tensor],
            principal_points_h: Union[float, torch.Tensor] = 0.0,
            principal_points_v: Union[float, torch.Tensor] = 0.0,
            focal_lengths: Union[float, torch.Tensor] = 970.0,
            ids: Union[None, List[None], int, List[int]] = None,
            multiplicity=1,
    ):
        """
        names: each camera has specific name. It can be str or a list of str in the case of batched camera
        screen_center_poses: It can be Tuple[float, float, float] or a torch.tensor with the shape of (batch_size, 3)
        screen_normals: It can be Tuple[float, float, float] or a torch.tensor with the shape of (batch_size, 3)
        screen_verticals: It can be Tuple[float, float, float] or a torch.tensor with the shape of (batch_size, 3)
        screen_sizes_h: in mm, It can be float or a torch.tensor with the shape of (batch_size, )
        screen_sizes_v: in mm, It can be float or a torch.tensor with the shape of (batch_size, )
        principal_points_h: in mm, It can be float or a torch.tensor with the shape of (batch_size, )
        principal_points_v: in mm, It can be float or a torch.tensor with the shape of (batch_size, )
        focal_lengths: in mm, It can be float or a torch.tensor with the shape of (batch_size, )
        ids: camera's id in database. You don't need to assign it when you have nothing to do with database
        """
        if not isinstance(names, list):
            names = [names]
        self.names = names
        self._N = len(names)  # I should check it if its type must be int or torch.Size()

        if not isinstance(ids, list):  # id in the database; i.e.: lmb_db.source_int_calibration.id.
            if (
                    ids is None
            ):  # this is for the moments you don't care about ids and want to create an object by default values
                ids = [None for i in range(self._N)]
            else:
                ids = [ids]
        assert len(ids) == self._N, "ids must have the same batch size as names has."
        self.ids = ids
        if not isinstance(screen_center_poses, torch.Tensor):
            screen_center_poses = torch.tensor(np.array([screen_center_poses]), dtype=DTYPE_TORCH)

        if not isinstance(screen_normals, torch.Tensor):
            screen_normals = torch.tensor(np.array([screen_normals]), dtype=DTYPE_TORCH)

        if not isinstance(screen_verticals, torch.Tensor):
            screen_verticals = torch.tensor(np.array([screen_verticals]), dtype=DTYPE_TORCH)

        if not isinstance(screen_sizes_h, torch.Tensor):
            screen_sizes_h = torch.tensor(np.array([screen_sizes_h]), dtype=DTYPE_TORCH)

        if not isinstance(screen_sizes_v, torch.Tensor):
            screen_sizes_v = torch.tensor(np.array([screen_sizes_v]), dtype=DTYPE_TORCH)

        if not isinstance(principal_points_h, torch.Tensor):
            principal_points_h = torch.tensor(np.array([principal_points_h]), dtype=DTYPE_TORCH)

        if not isinstance(principal_points_v, torch.Tensor):
            principal_points_v = torch.tensor(np.array([principal_points_v]), dtype=DTYPE_TORCH)

        if not isinstance(focal_lengths, torch.Tensor):
            focal_lengths = torch.tensor(np.array([focal_lengths]), dtype=DTYPE_TORCH)

        # breakpoint()
        assert screen_center_poses.shape == torch.Size([self._N, 3])
        assert screen_normals.shape == torch.Size([self._N, 3])
        assert screen_verticals.shape == torch.Size([self._N, 3])
        assert screen_sizes_h.shape == torch.Size([self._N])
        assert screen_sizes_v.shape == torch.Size([self._N])
        assert principal_points_h.shape == torch.Size([self._N])
        assert principal_points_v.shape == torch.Size([self._N])
        assert focal_lengths.shape == torch.Size([self._N])
        self.screen_center_poses = screen_center_poses.type(DTYPE_TORCH)
        self.screen_normals = torch.nn.functional.normalize(screen_normals, dim=1).type(DTYPE_TORCH)
        self.screen_verticals = torch.nn.functional.normalize(screen_verticals, dim=1).type(DTYPE_TORCH)

        # assert screen_verticals is perpendicular on screen_normals
        assert all(
            [
                torch.isclose(
                    torch.dot(self.screen_normals[i, :], self.screen_verticals[i, :]),
                    torch.tensor(0.0, dtype=DTYPE_TORCH),
                    atol=1e-04,
                )
                for i in range(self._N)
            ]
        ), "assert screen_verticals not perpendicular on screen_normals"
        self.screen_sizes_h = screen_sizes_h.type(DTYPE_TORCH)
        self.screen_sizes_v = screen_sizes_v.type(DTYPE_TORCH)
        # We keep the parameters in coordinate convection of opengl. But we change them when we want to use them
        self.pph = principal_points_h.type(DTYPE_TORCH)
        self.ppv = principal_points_v.type(DTYPE_TORCH)
        # remember that focal_length is the perpendicular distance from the screen to the camera position
        self.focal_lengths = focal_lengths
        self.pp_h_ndc = 2 * principal_points_h / screen_sizes_h
        self.pp_v_ndc = 2 * principal_points_v / screen_sizes_v

        # Note that matrix_view is calculated in coordinate system of opengl
        self.tmats = self.compute_tmat()  # camera to world
        tmats_inv = torch.linalg.inv(self.tmats)
        # X_cam = X_world R + T, so we need to transpose tmat_inv[:, 0:3, 0:3] to obtain R
        self.world_to_camera_T = tmats_inv[:, 0:3, 3]
        self.world_to_camera_R = tmats_inv[:, 0:3, 0:3].permute(0, 2, 1)

        field_of_view = 2 * torch.atan(screen_sizes_v / (2 * focal_lengths)) * 180 / np.pi
        h_to_v_ratio = screen_sizes_h / screen_sizes_v
        znear = self.focal_lengths / 10.0
        zfar = self.focal_lengths * 2
        FoVPerspectiveCameras.__init__(
            self,
            fov=field_of_view,
            aspect_ratio=h_to_v_ratio,
            znear=znear,
            zfar=zfar,
            R=self.world_to_camera_R,
            T=self.world_to_camera_T,
        )

    def compute_tmat(self) -> torch.Tensor:
        # dont look at the center of the xray screen but at the principal point
        # tmat is camera to world coordinate system (camera relative to world).
        # +z of camera is opposite of screen_normal.
        # +y of camera is the same as screen_vertical
        # +x of camera is obtained based +y and +z
        cam_pose = self.get_cam_pos()
        # target = self.get_cam_target()
        # up = self.screen_verticals
        yaxis = self.screen_verticals
        zaxis = -self.screen_normals
        # xaxis = torch.cross(up, zaxis, dim=1)  # it is already normalized
        xaxis = torch.cross(yaxis, zaxis, dim=1)  # it is already normalized
        # yaxis = torch.cross(zaxis, xaxis, dim=1)
        tmats = torch.eye(4, dtype=DTYPE_TORCH)[None].expand(self._N, 4, 4).clone()
        tmats[:, 0:3, 3] = cam_pose
        tmats[:, 0:3, 0] = xaxis
        tmats[:, 0:3, 1] = yaxis
        tmats[:, 0:3, 2] = zaxis
        return tmats

    def get_cam_pos(self):
        # position of the camera in world coordinate system
        z = self.focal_lengths.reshape(self._N, 1) * self.screen_normals
        # set z coordinate so that the distance along z between screen and eye is exactly the focal length
        cam_pos = self.get_cam_target() + z
        return cam_pos

    def get_cam_target(self):
        # position of the principal point in world coordinate system
        # note position of principal point in intensifier coordinate system (pph, ppv) has already determined by user
        # shift camera by principal point with respect to the screen
        h = self.pph.reshape(self._N, 1) * torch.cross(self.screen_verticals, self.screen_normals, dim=1)
        v = self.ppv.reshape(self._N, 1) * self.screen_verticals
        target = self.screen_center_poses + h + v
        return target

    def compute_projection_matrix(
            self, znear, zfar, fov, aspect_ratio, pp_h_ndc, pp_v_ndc, degrees
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)
        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            pp_h_ndc: principal_point in normal device coordinate. horizontal component
            pp_v_ndc: principal_point in normal device coordinate. vertical component
            degrees: bool, set to True if fov is specified in degrees.
        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((self._N, 4, 4), device=self.device, dtype=DTYPE_TORCH)
        ones = torch.ones(self._N, dtype=DTYPE_TORCH, device=self.device)
        if degrees:
            fov = (np.pi / 180) * fov

        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = -pp_h_ndc  # the minus sign is because of opposite directions for x axis
        K[:, 1, 2] = pp_v_ndc
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> "Transform3d":
        """
        Calculate the perspective projection matrix with a unsymmetric
        viewing frustrum. Use column major order.
        The viewing frustrum will be projected into ndc, s.t.
        (max_x, max_y) -> (+1, +1)
        (min_x, min_y) -> (-1, -1)
        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.
        Return:
            a Transform3d object which represents a batch of projection
            matrices of shape (N, 4, 4)
        .. code-block:: python
            h1 = (max_y + min_y)/(max_y - min_y) + pp_ndc_x
            w1 = (max_x + min_x)/(max_x - min_x) + pp_ndc_y
            tanhalffov = tan((fov/2))
            s1 = 1/tanhalffov
            s2 = 1/(tanhalffov * (aspect_ratio))
            # To map z to the range [0, 1] use:
            f1 =  far / (far - near)
            f2 = -(far * near) / (far - near)
            # Projection matrix
            K = [
                    [s1,   0,   w1,   0],
                    [0,   s2,   h1,   0],
                    [0,    0,   f1,  f2],
                    [0,    0,    1,   0],
            ]
        """
        from pytorch3d.transforms import Transform3d

        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % self._N)
        else:
            K = self.compute_projection_matrix(
                kwargs.get("znear", self.znear),
                kwargs.get("zfar", self.zfar),
                kwargs.get("fov", self.fov),
                kwargs.get("aspect_ratio", self.aspect_ratio),
                kwargs.get("pp_h_ndc", self.pp_h_ndc),
                kwargs.get("pp_v_ndc", self.pp_v_ndc),
                kwargs.get("degrees", self.degrees),
            )

        # Transpose the projection matrix as PyTorch3D transforms use row vectors.
        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform

    def get_batch_size(self):
        return self._N

    def copy(self, **kwarg):
        # to make a deep copy independent of previous camera
        copy_numbers = copy_names(self.names)

        camera_new = Camera(
            names=[
                self.names[i] + "_{}times_copy".format(copy_numbers[i] + 1)
                for i in range(self.get_batch_size())
            ],
            screen_center_poses=self.screen_center_poses.clone(),
            screen_normals=self.screen_normals.clone(),
            screen_verticals=self.screen_verticals.clone(),
            screen_sizes_h=self.screen_sizes_h.clone(),
            screen_sizes_v=self.screen_sizes_v.clone(),
            principal_points_h=self.pph.clone(),
            principal_points_v=self.ppv.clone(),
            focal_lengths=self.focal_lengths.clone(),
            ids=self.ids,
        )

        return camera_new

    def to(self, new_device: Union[str, torch.device]):
        """
        transfer all torch.tensor attributes to new_device
        it changes self.
        if new_device is the same with old_device, it does nothing
        """
        old_device = self.device
        new_device = determine_device(new_device)
        if not new_device == old_device:
            for k in dir(self):
                v = getattr(self, k)
                if k == "device":
                    setattr(self, k, new_device)
                if torch.is_tensor(v):
                    setattr(self, k, v.to(new_device))
        return self


def join_cameras_as_batch(cameras: List[Camera]):
    cameras = [camera.to("cpu") for camera in cameras]
    camera_new = Camera(
        names=[camera.names for camera in cameras],
        screen_center_poses=torch.cat([camera.screen_center_poses for camera in cameras], dim=0),
        screen_normals=torch.cat([camera.screen_normals for camera in cameras], dim=0),
        screen_verticals=torch.cat([camera.screen_verticals for camera in cameras], dim=0),
        screen_sizes_h=torch.cat([camera.screen_sizes_h for camera in cameras], dim=0),
        screen_sizes_v=torch.cat([camera.screen_sizes_v for camera in cameras], dim=0),
        principal_points_h=torch.cat([camera.pph for camera in cameras], dim=0),
        principal_points_v=torch.cat([camera.ppv for camera in cameras], dim=0),
        focal_lengths=torch.cat([camera.focal_lengths for camera in cameras], dim=0),
        ids=[id for camera in cameras for id in camera.ids],
    )
    return camera_new


def create_batched_actors_from_single_actor(actor: Union[AnatomySTL, AnatomyCT, Camera], bath_size=1):
    actors = [actor]
    for i in range(bath_size - 1):
        actors.append(actors[i].copy())

    if isinstance(actor, AnatomySTL):
        return join_anatomies_stl_as_batch(anatomies=actors)
    elif isinstance(actor, AnatomyCT):
        return join_anatomies_ct_as_batch(anatomies=actors)
    elif isinstance(actor, Camera):
        return join_cameras_as_batch(cameras=actors)
    else:
        raise TypeError("actor must belong to AnatomySTL, AnatomyCT or Camera")


class CylinderSTL(AnatomySTL):
    """
    This class is just for creating a simple cylinder for testing the renderer.
    """

    def __init__(
            self,
            diameter: float = 50.0,
            depth: float = 50.0,
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size: int = 1,
    ):
        """
        All the input arguments are in mm
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        """
        self.diameter = diameter
        self.depth = depth
        self.center_pose = center_pose
        mesh = self.create_mesh(diameter, depth, center_pose, 100)
        super(CylinderSTL, self).__init__(
            names=["simple_stl_cylinder_{}".format(i) for i in range(batch_size)],
            verts=[mesh["verts"] for i in range(batch_size)],
            faces=[mesh["faces"] for i in range(batch_size)],
            verts_normals=[mesh["verts_normals"] for i in range(batch_size)],
        )

    @staticmethod
    def create_mesh(diameter, depth, center_pose, nfrags):
        verts = torch.zeros((3 * 4 * nfrags, 3), dtype=DTYPE_TORCH)
        verts_normals = torch.zeros((3 * 4 * nfrags, 3), dtype=DTYPE_TORCH)
        t = torch.linspace(0, 2 * np.pi, nfrags + 1).type(DTYPE_TORCH)
        x = diameter / 2.0 * torch.cos(t)
        y = diameter / 2.0 * torch.sin(t)
        d = depth / 2.0
        center = torch.tensor(center_pose, dtype=DTYPE_TORCH)
        for i in range(nfrags):
            o1 = torch.tensor([0, 0, d], dtype=DTYPE_TORCH)
            a1 = torch.tensor([x[i], y[i], d], dtype=DTYPE_TORCH)
            b1 = torch.tensor([x[i + 1], y[i + 1], d], dtype=DTYPE_TORCH)
            o2 = torch.tensor([0, 0, -d], dtype=DTYPE_TORCH)
            a2 = torch.tensor([x[i], y[i], -d], dtype=DTYPE_TORCH)
            b2 = torch.tensor([x[i + 1], y[i + 1], -d], dtype=DTYPE_TORCH)

            verts[12 * i + 0, :] = o1
            verts[12 * i + 1, :] = a1
            verts[12 * i + 2, :] = b1
            verts_normals[12 * i + 0, :] = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE_TORCH)
            verts_normals[12 * i + 1, :] = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE_TORCH)
            verts_normals[12 * i + 2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE_TORCH)

            verts[12 * i + 3, :] = o2
            verts[12 * i + 4, :] = a2
            verts[12 * i + 5, :] = b2
            verts_normals[12 * i + 3, :] = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE_TORCH)
            verts_normals[12 * i + 4, :] = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE_TORCH)
            verts_normals[12 * i + 5, :] = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE_TORCH)

            verts[12 * i + 6, :] = a1
            verts[12 * i + 7, :] = a2
            verts[12 * i + 8, :] = b1
            verts_normals[12 * i + 6, :] = torch.nn.functional.normalize(a1 - o1, dim=0)
            verts_normals[12 * i + 7, :] = torch.nn.functional.normalize(a1 - o1, dim=0)
            verts_normals[12 * i + 8, :] = torch.nn.functional.normalize(a1 - o1, dim=0)

            verts[12 * i + 9, :] = b1
            verts[12 * i + 10, :] = a2
            verts[12 * i + 11, :] = b2
            verts_normals[12 * i + 9, :] = torch.nn.functional.normalize(a2 - o2, dim=0)
            verts_normals[12 * i + 10, :] = torch.nn.functional.normalize(a2 - o2, dim=0)
            verts_normals[12 * i + 11, :] = torch.nn.functional.normalize(a2 - o2, dim=0)

        faces = torch.tensor(
            [
                [i * 3, i * 3 + 1, i * 3 + 2]
                for i in range(torch.div(verts.shape[0], 3, rounding_mode="floor"))
            ],
            dtype=DTYPE_TORCH,
        )

        return {"verts": verts + center, "faces": faces, "verts_normals": verts_normals}


class SphereSTL(AnatomySTL):
    """
    This class is just for creating a simple sphere for testing the renderer.
    """

    def __init__(
            self,
            diameter: float = 50.0,
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size: int = 1,
    ):
        """
        All the input arguments are in mm
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        """
        self.diameter = diameter
        self.center_pose = center_pose
        mesh = self.create_mesh(diameter / 2, center_pose, 100)
        super(SphereSTL, self).__init__(
            names=["simple_stl_sphere_{}".format(i) for i in range(batch_size)],
            verts=[mesh["verts"] for i in range(batch_size)],
            faces=[mesh["faces"] for i in range(batch_size)],
            verts_normals=[mesh["verts_normals"] for i in range(batch_size)],
        )

    @staticmethod
    def create_mesh(radius, center_pose, nfrags):
        theta = torch.linspace(-np.pi, np.pi, nfrags)[None]
        phi = (torch.linspace(-np.pi / 2, np.pi / 2, nfrags)[None]).permute(1, 0)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        xx = radius * cosphi * costheta
        yy = radius * cosphi * sintheta
        zz = radius * sinphi * torch.ones_like(theta)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        center = torch.tensor(center_pose, dtype=DTYPE_TORCH)
        verts = torch.tensor((), dtype=DTYPE_TORCH)
        verts_normals = torch.tensor((), dtype=DTYPE_TORCH)
        for i in range(nfrags ** 2):
            f1 = [i, i + nfrags, i + 1]
            f2 = [i + nfrags, i + nfrags + 1, i + 1]
            for _f in [f1, f2]:
                try:
                    p1 = torch.tensor([xx[_f[0]], yy[_f[0]], zz[_f[0]]], dtype=DTYPE_TORCH)
                    p2 = torch.tensor([xx[_f[1]], yy[_f[1]], zz[_f[1]]], dtype=DTYPE_TORCH)
                    p3 = torch.tensor([xx[_f[2]], yy[_f[2]], zz[_f[2]]], dtype=DTYPE_TORCH)
                    nvec = -torch.nn.functional.normalize(torch.cross(p2 - p1, p3 - p1, dim=0), dim=0)
                    verts = torch.cat((verts, p1[None], p2[None], p3[None]), dim=0)
                    verts_normals = torch.cat((verts_normals, nvec[None], nvec[None], nvec[None]), dim=0)

                except IndexError:
                    pass

        faces = torch.tensor(
            [
                [i * 3, i * 3 + 1, i * 3 + 2]
                for i in range(torch.div(verts.shape[0], 3, rounding_mode="floor"))
            ],
            dtype=DTYPE_TORCH,
        )

        return {"verts": verts + center, "faces": faces, "verts_normals": verts_normals}


class CubeSTL(AnatomySTL):
    """
    This class is just for creating a simple cube for testing the renderer.
    """

    def __init__(
            self,
            width: float = 50.0,
            height: float = 50.0,
            depth: float = 50.0,
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size=1,
    ):
        """
        All the input arguments are in mm
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        """

        self.width = width
        self.height = height
        self.depth = depth
        self.center_pose = center_pose
        mesh = self.create_mesh(width, height, depth, center_pose)
        super(CubeSTL, self).__init__(
            names=["simple_stl_cube_{}".format(i) for i in range(batch_size)],
            verts=[mesh["verts"] for i in range(batch_size)],
            faces=[mesh["faces"] for i in range(batch_size)],
            verts_normals=[mesh["verts_normals"] for i in range(batch_size)],
        )

    @staticmethod
    def create_mesh(width, height, depth, center_pose):
        w = width / 2.0
        h = height / 2.0
        d = depth / 2.0
        center = torch.tensor(center_pose, dtype=DTYPE_TORCH)
        verts_and_normals = torch.tensor(
            [
                -w,
                -h,
                d,
                0,
                0,
                1,
                w,
                -h,
                d,
                0,
                0,
                1,
                w,
                h,
                d,
                0,
                0,
                1,
                -w,
                -h,
                d,
                0,
                0,
                1,
                w,
                h,
                d,
                0,
                0,
                1,
                -w,
                h,
                d,
                0,
                0,
                1,
                -w,
                -h,
                -d,
                0,
                0,
                -1,
                w,
                -h,
                -d,
                0,
                0,
                -1,
                w,
                h,
                -d,
                0,
                0,
                -1,
                -w,
                -h,
                -d,
                0,
                0,
                -1,
                w,
                h,
                -d,
                0,
                0,
                -1,
                -w,
                h,
                -d,
                0,
                0,
                -1,
                -w,
                -h,
                d,
                0,
                -1,
                0,
                w,
                -h,
                d,
                0,
                -1,
                0,
                w,
                -h,
                -d,
                0,
                -1,
                0,
                -w,
                -h,
                d,
                0,
                -1,
                0,
                w,
                -h,
                -d,
                0,
                -1,
                0,
                -w,
                -h,
                -d,
                0,
                -1,
                0,
                w,
                -h,
                d,
                1,
                0,
                0,
                w,
                h,
                d,
                1,
                0,
                0,
                w,
                -h,
                -d,
                1,
                0,
                0,
                w,
                h,
                d,
                1,
                0,
                0,
                w,
                -h,
                -d,
                1,
                0,
                0,
                w,
                h,
                -d,
                1,
                0,
                0,
                w,
                h,
                d,
                0,
                1,
                0,
                -w,
                h,
                d,
                0,
                1,
                0,
                w,
                h,
                -d,
                0,
                1,
                0,
                -w,
                h,
                d,
                0,
                1,
                0,
                w,
                h,
                -d,
                0,
                1,
                0,
                -w,
                h,
                -d,
                0,
                1,
                0,
                -w,
                -h,
                d,
                -1,
                0,
                0,
                -w,
                h,
                d,
                -1,
                0,
                0,
                -w,
                h,
                -d,
                -1,
                0,
                0,
                -w,
                -h,
                d,
                -1,
                0,
                0,
                -w,
                -h,
                -d,
                -1,
                0,
                0,
                -w,
                h,
                -d,
                -1,
                0,
                0,
            ],
            dtype=DTYPE_TORCH,
        ).reshape((36, 6))
        faces = torch.tensor(
            [
                [i * 3, i * 3 + 1, i * 3 + 2]
                for i in range(torch.div(verts_and_normals.shape[0], 3, rounding_mode="floor"))
            ],
            dtype=DTYPE_TORCH,
        )

        return {
            "verts": verts_and_normals[:, 0:3] + center,
            "faces": faces,
            "verts_normals": verts_and_normals[:, 3:6],
        }


class CylinderCT(AnatomyCT):
    """
    This class is just for creating a simple cylinder for testing the renderer.
    It creates a cube having a cylinder in itself.
    The orientation of the cylinder is like that the cylinder is perpendicular to the world coord's x-y plane.
    The voxels inside of the cylinder have a intensity equal to 1.
    The voxels outside of the cylinder have a intensity equal to 0.
    """

    def __init__(
            self,
            diameter: float = 50.0,
            depth: float = 50.0,
            grid_size: Tuple[int, int, int] = (50, 50, 50),
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size: int = 1,
    ):
        """
        diameter, depth and center_pose are in mm.
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        grid_size is in voxel, and it is in this order (depth:z, height: y, width: x)
        It determines the cube's size containing the cylinder.
        """
        self.diameter = diameter
        self.depth = depth
        densities, intrinsic_affine_mat = self.create_voxel_grid(diameter, depth, grid_size, center_pose)
        super(CylinderCT, self).__init__(
            names=["simple_ct_cylinder_{}".format(i) for i in range(batch_size)],
            densities=[densities for i in range(batch_size)],
            intrinsic_affine_mats=[intrinsic_affine_mat for i in range(batch_size)],
        )

    @staticmethod
    def create_voxel_grid(diameter, depth, grid_size, center_pose):
        # voxel_size is in the x, y, z order, but grid_size is in z, y, x order
        voxel_size = torch.tensor([diameter, diameter, depth], dtype=DTYPE_TORCH) / torch.tensor(
            [grid_size[2], grid_size[1], grid_size[0]], dtype=DTYPE_TORCH
        )
        radius = diameter / 2.0
        x = torch.linspace(-radius + voxel_size[0] / 2, radius - voxel_size[0] / 2, grid_size[2])
        y = torch.linspace(-radius + voxel_size[1] / 2, radius - voxel_size[1] / 2, grid_size[1])
        z = torch.linspace(-depth / 2 + voxel_size[2] / 2, depth / 2 - voxel_size[2] / 2, grid_size[0])
        zz, yy, xx = torch.meshgrid(z, y, x)
        rr = torch.sqrt(torch.pow(xx, 2) + torch.pow(yy, 2))
        densities = torch.zeros(*grid_size, dtype=DTYPE_TORCH)
        densities[rr < radius] = 1.0

        volume_translation = torch.eye(4, dtype=DTYPE_TORCH)[None]
        volume_translation[0, 0:3, 3] = torch.tensor(center_pose, dtype=DTYPE_TORCH) + torch.tensor(
            [-radius + voxel_size[0] / 2, -radius + voxel_size[1] / 2, -depth / 2 + voxel_size[2] / 2]
        )
        volume_scale = torch.eye(4, dtype=DTYPE_TORCH)[None]
        diag_elem = torch.eye(3, dtype=bool)
        volume_scale[0, 0:3, 0:3][diag_elem] = voxel_size
        intrinsic_affine_mat = torch.matmul(volume_translation, volume_scale)
        return densities[None, None], intrinsic_affine_mat


class SphereCT(AnatomyCT):
    """
    This class is just for creating a simple sphere for testing the renderer.
    It creates a cube having a sphere in itself.
    The voxels inside of the sphere have a intensity equal to 1.
    The voxels outside of the sphere have a intensity equal to 0.
    """

    def __init__(
            self,
            diameter: float = 50.0,
            grid_size: Tuple[int, int, int] = (50, 50, 50),
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size: int = 1,
            voxel_density=1.0,
    ):
        """
        diameter and center_pose are in mm.
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        grid_size is in voxel, and it is in this order (depth:z, height: y, width: x).
        It determines the cube's size containing the sphere.
        """
        self.diameter = diameter
        densities, intrinsic_affine_mat = self.create_voxel_grid(diameter, grid_size, center_pose, voxel_density)
        super(SphereCT, self).__init__(
            names=["simple_ct_sphere_{}".format(i) for i in range(batch_size)],
            densities=[densities for i in range(batch_size)],
            intrinsic_affine_mats=[intrinsic_affine_mat for i in range(batch_size)],
        )

    @staticmethod
    def create_voxel_grid(diameter, grid_size, center_pose, voxel_density):
        # voxel_size is in the x, y, z order, but grid_size is in z, y, x order
        voxel_size = torch.tensor([diameter, diameter, diameter], dtype=DTYPE_TORCH) / torch.tensor(
            [grid_size[2], grid_size[1], grid_size[0]], dtype=DTYPE_TORCH
        )
        radius = diameter / 2.0
        x = torch.linspace(-radius + voxel_size[0] / 2, radius - voxel_size[0] / 2, grid_size[2])
        y = torch.linspace(-radius + voxel_size[1] / 2, radius - voxel_size[1] / 2, grid_size[1])
        z = torch.linspace(-radius + voxel_size[2] / 2, radius - voxel_size[2] / 2, grid_size[0])
        zz, yy, xx = torch.meshgrid(z, y, x)
        rr = torch.sqrt(torch.pow(xx, 2) + torch.pow(yy, 2) + torch.pow(zz, 2))
        densities = torch.zeros(*grid_size, dtype=DTYPE_TORCH)
        densities[rr < radius] = voxel_density

        # volume_translation is correspoding to location of voxel(0, 0, 0) in the world coordinate system
        volume_translation = torch.eye(4, dtype=DTYPE_TORCH)[None]
        volume_translation[0, 0:3, 3] = torch.tensor(center_pose, dtype=DTYPE_TORCH) + torch.tensor(
            [-radius + voxel_size[0] / 2, -radius + voxel_size[1] / 2, -radius + voxel_size[2] / 2]
        )

        volume_scale = torch.eye(4, dtype=DTYPE_TORCH)[None]
        diag_elem = torch.eye(3, dtype=bool)
        volume_scale[0, 0:3, 0:3][diag_elem] = voxel_size
        intrinsic_affine_mat = torch.matmul(volume_translation, volume_scale)

        return densities[None, None], intrinsic_affine_mat


class CubeCT(AnatomyCT):
    """
    This class is just for creating a simple sphere for testing the renderer.
    All the voxels have the value equal to 1.
    """

    def __init__(
            self,
            width: float = 50.0,
            height: float = 50.0,
            depth: float = 50.0,
            grid_size: Tuple[int, int, int] = (50, 50, 50),
            center_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            batch_size=1,
            voxel_density=1.0,
    ):
        """
        width, height, depth and center_pose are in millimeter
        center_pose is in this order: (x, y, z), and it must be defined w.r.t your world coordinate system.
        grid_size is in voxel, and it is in this order (depth:z, height: y, width: x).
        """
        self.width = width
        self.height = height
        self.depth = depth
        densities, intrinsic_affine_mat = self.create_voxel_grid(width, height, depth, grid_size,
                                                                 center_pose, voxel_density)
        super(CubeCT, self).__init__(
            names=["simple_ct_cube_{}".format(i) for i in range(batch_size)],
            densities=[densities for i in range(batch_size)],
            intrinsic_affine_mats=[intrinsic_affine_mat for i in range(batch_size)],
        )

    @staticmethod
    def create_voxel_grid(width, height, depth, grid_size, center_pose, voxel_density):
        # voxel_size is in the x, y, z order, but grid_size is in z, y, x order
        voxel_size = torch.tensor([width, height, depth], dtype=DTYPE_TORCH) / torch.tensor(
            [grid_size[2], grid_size[1], grid_size[0]], dtype=DTYPE_TORCH
        )

        densities = voxel_density * torch.ones(*grid_size, dtype=DTYPE_TORCH)

        volume_translation = torch.eye(4, dtype=DTYPE_TORCH)[None]
        volume_translation[0, 0:3, 3] = torch.tensor(center_pose, dtype=DTYPE_TORCH) + torch.tensor(
            [-width / 2 + voxel_size[0] / 2, -height / 2 + voxel_size[1] / 2, -depth / 2 + voxel_size[2] / 2]
        )
        volume_scale = torch.eye(4, dtype=DTYPE_TORCH)[None]
        diag_elem = torch.eye(3, dtype=bool)
        volume_scale[0, 0:3, 0:3][diag_elem] = voxel_size
        intrinsic_affine_mat = torch.matmul(volume_translation, volume_scale)
        return densities[None, None], intrinsic_affine_mat


class Scene:
    """
    Not that if you are going to use this class as a batched scenes(multi frames),
    all the objects must have same batch size
    """

    def __init__(
            self,
            anatomies: Union[AnatomySTL, AnatomyCT, List[AnatomySTL], List[AnatomyCT]] = [],
            cameras: Union[Camera, List[Camera]] = [],
    ):
        """
        You can determine your anatomies and cameras when you are creating your scene, or
        you can add them after you create the scene.

        you can access to an anatomy or camera by its corresponding index: your_scene.anatomies[i].
        When you add a anatomy or camera to a scene, it gets an index depending on the order you added it.
        I recommend that when you want to create a scene, create also a dictionary like this:
        anatomies_indices = {"femur": 0, "tibia": 1}.
        Be careful the number related to each anatomy be the same as the order that anatomy added to your scene.
        so you can access to the femur through:  your_scene.anatomies[anatomies_indices["femur"]]

        Same thing for the camera if your scene has multiple cameras.
        These dictionary help you to know index of each anatomy or camera.
        """
        self.anatomies = []
        self.cameras = []
        self._N = 0
        self.anatomies_type = ""
        self.add_anatomies(anatomies)
        self.add_cameras(cameras)

    def check_validity_anatomies(self, anatomies):
        """
        anatomies: it can be single anatomy with arbitrary batch_size or a list of anatomies with the same batch_size

        check to see if all the anatomies belong to the same class(AnatomyCT or AnatomySTL),
        and check to see if all the anatomies have the same batch_size
        """
        if not isinstance(anatomies, list):
            anatomies = [anatomies]

        if (self.anatomies_type == "") and (anatomies != []):  # if scenes has not already any anatomy
            self.anatomies_type = "CT" if isinstance(anatomies[0], AnatomyCT) else "STL"
        for anatomy in anatomies:
            if self.anatomies_type == "CT":
                assert isinstance(anatomy, AnatomyCT), "all anatomies in the input must belong to same class"
            elif self.anatomies_type == "STL":
                assert isinstance(anatomy, AnatomySTL), "all anatomies in the input must belong to same class"

        if (self._N == 0) and (anatomies != []):  # check if scenes already have actor or not
            self._N = anatomies[0].get_batch_size()

        for anatomy in anatomies:
            assert (
                anatomy.get_batch_size() == self._N
            ), "all objects in the scenes must have the same batch_size"

    def check_validity_cameras(self, cameras):
        """
        cameras: it can be single camera with arbitrary batch_size or a list of cameras with the same batch_size

        Check to see if all the cameras have the same batch_size
        """
        if not isinstance(cameras, list):
            cameras = [cameras]

        if (self._N == 0) and (cameras != []):  # check if scenes already have actor or not
            self._N = cameras[0].get_batch_size()

        for camera in cameras:
            assert (
                camera.get_batch_size() == self._N
            ), "all objects in the scenes must have the same batch_size"

    def add_anatomies(self, anatomies: Union[AnatomySTL, AnatomyCT, List[AnatomySTL], List[AnatomyCT]]):
        """
        anatomies: it can be single anatomy with arbitrary batch_size,
         or a list of anatomies of the same type with the same batch_size
        """
        self.check_validity_anatomies(anatomies=anatomies)
        if not isinstance(anatomies, list):
            anatomies = [anatomies]
        for i in range(len(anatomies)):
            self.anatomies.append(anatomies[i])

    def add_cameras(self, cameras: Union[Camera, List[Camera]]):
        """
        cameras: it can be single camera with arbitrary batch_size,
         or a list of camera with the same batch_size
        """
        if not cameras:
            return
        self.check_validity_cameras(cameras=cameras)
        if not isinstance(cameras, list):
            cameras = [cameras]
        for i in range(len(cameras)):
            self.cameras.append(cameras[i])

    def del_anatomies(self, indices: Union[int, List[int]]):
        """
        you can delete an anatomy or several anatomies from your scene by knowing their index.
        """
        if not isinstance(indices, list):
            indices = [indices]
        assert len(indices) <= len(self.anatomies)
        assert all(
            [(indices[i] >= 0) and (indices[i] < len(indices)) for i in range(len(indices))]
        ), " indices must be between zero and {}".format(len(indices))
        for index in sorted(indices, reverse=True):
            del self.anatomies[index]

    def del_cameras(self, indices: Union[int, List[int]]):
        """
        you can delete an camera or several cameras from your scene by knowing their index.
        """
        if not isinstance(indices, list):
            indices = [indices]
        assert len(indices) <= len(self.cameras)
        assert all(
            [(indices[i] >= 0) and (indices[i] < len(indices)) for i in range(len(indices))]
        ), " indices must be between zero and {}".format(len(indices))
        for index in sorted(indices, reverse=True):
            del self.cameras[index]

    def get_batch_size(self):
        return self._N

    def to(self, new_device: Union[str, torch.device]):
        """
        transfer all the anatomies and cameras of the scene to a new device, and return itself
        """
        self.anatomies = [anatomy.to(new_device) for anatomy in self.anatomies]
        self.cameras = [camera.to(new_device) for camera in self.cameras]
        return self


def join_scenes_as_batch(scenes: List[Scene]):
    anatomies_type = scenes[0].anatomies_type
    anatomies = []
    cameras = []
    for scene in scenes:
        assert scene.anatomies_type == anatomies_type, "all scenes must have the same anatomies_type"
    for scene in scenes:
        assert len(scene.anatomies) == len(scenes[0].anatomies), "all scenes must have the anatomy number"
    for scene in scenes:
        assert len(scene.cameras) == len(scenes[0].cameras), "all scenes must have the camera number"
    if anatomies_type == "CT":
        for i in range(len(scenes[0].anatomies)):
            anatomies.append(join_anatomies_ct_as_batch(anatomies=[scene.anatomies[i] for scene in scenes]))

    elif anatomies_type == "STL":
        for i in range(len(scenes[0].anatomies)):
            anatomies.append(join_anatomies_stl_as_batch(anatomies=[scene.anatomies[i] for scene in scenes]))

    for i in range(len(scenes[0].cameras)):
        cameras.append(join_cameras_as_batch(cameras=[scene.cameras[i] for scene in scenes]))

    return Scene(anatomies=anatomies, cameras=cameras)
