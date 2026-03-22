from typing import TypeAlias

from schemas.overridable import OverridableModel
from .enums import AlphaMode

class GLBConverterParams(OverridableModel):
    """GLB conversion parameters with automatic fallback to settings."""
    decimation_target: int 
    texture_size: int 
    alpha_mode: AlphaMode 
    rescale: float 
    remesh: bool
    remesh_band: float 
    remesh_project: float 
    mesh_cluster_refine_iterations: int 
    mesh_cluster_global_iterations: int 
    mesh_cluster_smooth_strength: float 
    mesh_cluster_threshold_cone_half_angle: float
    subdivisions: int
    vertex_reproject: float
    alpha_gamma: float
    
    
    @classmethod
    def from_settings(cls, settings) -> "GLBConverterParams":
        return cls(
            decimation_target=settings.decimation_target,
            texture_size=settings.texture_size,
            alpha_mode=settings.alpha_mode,
            rescale=settings.rescale,
            remesh=settings.remesh,
            remesh_band=settings.remesh_band,
            remesh_project=settings.remesh_project,
            mesh_cluster_refine_iterations=settings.mesh_cluster_refine_iterations,
            mesh_cluster_global_iterations=settings.mesh_cluster_global_iterations,
            mesh_cluster_smooth_strength=settings.mesh_cluster_smooth_strength,
            mesh_cluster_threshold_cone_half_angle=settings.mesh_cluster_threshold_cone_half_angle,
            subdivisions=settings.subdivisions,
            vertex_reproject=settings.vertex_reproject,
            alpha_gamma=settings.alpha_gamma,
        )


GLBConverterParamsOverrides: TypeAlias = GLBConverterParams.Overrides
