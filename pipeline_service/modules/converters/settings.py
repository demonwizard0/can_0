from pydantic import BaseModel
from .enums import AlphaMode


class GLBConverterConfig(BaseModel):
    """GLB converter configuration"""
    decimation_target: int = 1000000
    texture_size: int = 1024
    alpha_mode: AlphaMode = AlphaMode.OPAQUE
    rescale: float = 1.0
    remesh: bool = True
    remesh_band: float = 1.0
    remesh_project: float = 0.0
    mesh_cluster_refine_iterations: int = 0
    mesh_cluster_global_iterations: int = 1
    mesh_cluster_smooth_strength: float = 1.0
    mesh_cluster_threshold_cone_half_angle: float = 90.0
    subdivisions: int = 2
    vertex_reproject: float = 0.0
    alpha_gamma: float = 2.2
    gpu: int = 0
