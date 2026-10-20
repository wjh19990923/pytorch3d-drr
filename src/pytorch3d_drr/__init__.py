"""Lazy public API; diagnostics work without rendering dependencies."""
from importlib import import_module

__version__ = "0.1.0.dev0"
_exports = {
    "pytorch3d_renderer": ("CTRenderer", "STLRenderer"),
    "raymarching_generalized": ("AbsorptionOnlyRaymarcherGeneralized",),
    "raysampling_generalized": ("GridRaysamplerGeneralized", "NDCGridRaysamplerGeneralized", "PartialNDCGridRaysampler"),
    "scene_tools": (
        "DTYPE_TORCH", "AnatomyCT", "AnatomySTL", "Camera", "CubeCT", "CubeSTL",
        "CylinderCT", "CylinderSTL", "ModelBase", "Scene", "SphereCT", "SphereSTL",
        "anatomy_ct_to_anatomy_stl", "create_batched_actors_from_single_actor",
        "join_anatomies_ct_as_batch", "join_anatomies_stl_as_batch",
        "join_cameras_as_batch", "join_scenes_as_batch"),
    "utilities": ("determine_device",),
}
__all__ = [name for names in _exports.values() for name in names]


def __getattr__(name):
    for module, names in _exports.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
