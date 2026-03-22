from typing import Optional

from pydantic import BaseModel

from schemas.enums import PromptType
from modules.mesh_generator.schemas import TrellisParamsOverrides
from modules.converters.params import GLBConverterParamsOverrides

class GenerationRequest(BaseModel):
    # Prompt data
    prompt_type: PromptType = PromptType.IMAGE
    prompt_image: str 
    seed: int = -1
    render_grid_view: bool = False

    # Trellis parameters
    trellis_params: Optional[TrellisParamsOverrides] = None
    glbconv_params: Optional[GLBConverterParamsOverrides] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt_type": "text",
                "prompt_image": "file_name.jpg",
                "seed": 42,
                "render_grid_view": False,
                "trellis_params": {
                    "sparse_structure_steps": 12,
                    "sparse_structure_cfg_strength": 7.5,
                    "shape_slat_steps": 12,
                    "shape_slat_cfg_strength": 7.5,
                    "tex_slat_steps": 12,
                    "tex_slat_cfg_strength": 7.5,
                    "pipeline_type": "1024_cascade",
                    "max_num_tokens": 49152,
                },
                "glbconv_params": {
                    "decimation_target": 1000,
                    "texture_size": 512,
                    "alpha_mode": "OPAQUE",
                    "rescale": 1.0,
                    "remesh": True,
                    "remesh_band": 1.0,
                    "remesh_project": 0.0,
                    "mesh_cluster_refine_iterations": 0,
                    "mesh_cluster_global_iterations": 1,
                    "mesh_cluster_smooth_strength": 1.0,
                    "mesh_cluster_threshold_cone_half_angle": 90.0,
                }
            }
        }

